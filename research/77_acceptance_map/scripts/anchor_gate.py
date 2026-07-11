#!/usr/bin/env python3
"""Phase 77 anchor gate (HARD, run before any sweep cell).

Measures OFFLINE teacher-forced window-draft acceptance beta on the two cells
where 76-E3 measured real end-to-end accept lengths, composes geometric tau,
and demands they agree:

  dense Qwen2.5-7B  ctx16k window512+sinks16  K=4 -> E3 accept 4.850
  MoE  Qwen3-30B    ctx32k window512+sinks16  K=6 -> E3 accept 6.493

Method (the phase's primitive, exercised end-to-end here):
  1. Build W7-style prompts (filler para + doc marker + question + chat
     template) at ctx tokens -- same text distribution as E3.
  2. Target: chunked prefill (full attention, KV cached), then GREEDY generate
     G tokens step by step (E3 ran greedy).
  3. At every generated position: a DRAFT decode step re-feeds the last token
     against the target KV SLICED to sinks(16) + window(512) -- keys keep their
     original RoPE (vLLM window draft attends sink pages + tail, no
     re-rotation), the query is rotated at its true position. Records the
     greedy top-1 match and the T=1 acceptance overlap sum min(p_d, p_t).
  4. beta = mean top-1 match over generated positions;
     tau_geom(K) = (1 - beta^(K+1)) / (1 - beta); compare vs E3.

Depth>0 decay is NOT modeled (P24: geometric is exact to k<=4, ~0.97 at k=6-8),
so a small OVER-prediction at K=6 is expected and tolerated:
PASS <=3% | SOFT 3-6% (document) | FAIL >6% (method miscalibrated, stop).

Usage: python anchor_gate.py --cell dense|moe [--prompts 4 --gen 96]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

PHASE = Path(__file__).resolve().parent.parent
REPO = PHASE.parent.parent
PROMPTS = REPO / "research/57_large_ep_spec_strategy/data/prompts_ondist.txt"

CELLS = {
    "dense": dict(model="Qwen/Qwen2.5-7B-Instruct", ctx=16384, K=4, e3=4.850,
                  trust=False),
    "moe": dict(model="Qwen/Qwen3-30B-A3B", ctx=32768, K=6, e3=6.493,
                trust=False),
}

FILLER = (
    "In the broader study of natural language and machine reasoning, "
    "researchers have long observed that context shapes meaning in subtle "
    "and far-reaching ways, and that the surrounding passage a model reads "
    "before it answers can change every prediction that follows. The city "
    "sat at the edge of a wide river, and each morning the markets filled "
    "with traders carrying grain, salt, cloth, and stories gathered from "
    "distant provinces. Over the centuries, scholars debated the nature of "
    "memory, the structure of language, the movement of the planets, and "
    "the slow accumulation of knowledge that turns observation into theory. "
    "A traveller who kept a careful journal recorded the weather, the price "
    "of bread, the names of the ships in the harbour, and the arguments of "
    "the philosophers who gathered in the shaded courtyards to reason about "
    "cause and consequence, about what can be known and what must be "
    "supposed. These records, though ordinary in their day, later became a "
    "window into a vanished world of commerce, curiosity, and quiet labor. "
)


def build_prompts(tok, ctx_tokens: int, n: int) -> list[str]:
    """W7-faithful: filler(rotated, marker) + question, chat template."""
    bank = [ln.strip() for ln in PROMPTS.read_text().splitlines() if ln.strip()][:n]
    corpus = tok(FILLER, add_special_tokens=False).input_ids
    while len(corpus) < 2 * ctx_tokens + 64:
        corpus = corpus + corpus
    out = []
    for i, p in enumerate(bank):
        p_ids = tok(p, add_special_tokens=False).input_ids
        marker = f"Document #{i}: "
        m_ids = tok(marker, add_special_tokens=False).input_ids
        need = max(0, ctx_tokens - len(p_ids) - len(m_ids))
        off = (i * 997) % (len(corpus) - need) if need > 0 else 0
        fill = tok.decode(corpus[off:off + need])
        text = f"{marker}{fill}\n\nNow answer this question. {p}"
        out.append(tok.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True, tokenize=False))
    return out


def cache_kv(cache, layer: int):
    """(k, v) for a layer across transformers cache APIs."""
    if hasattr(cache, "layers"):                       # transformers 5.x
        lay = cache.layers[layer]
        return lay.keys, lay.values
    return cache.key_cache[layer], cache.value_cache[layer]  # 4.x


def sliced_cache(cache, n_layers: int, upto: int, sinks: int, window: int):
    """New cache holding target KV positions [0:sinks] + [upto-window:upto]."""
    from transformers import DynamicCache
    new = DynamicCache()
    lo = max(sinks, upto - window)
    for li in range(n_layers):
        k, v = cache_kv(cache, li)
        ks = torch.cat([k[:, :, :sinks], k[:, :, lo:upto]], dim=2)
        vs = torch.cat([v[:, :, :sinks], v[:, :, lo:upto]], dim=2)
        new.update(ks, vs, li)
    return new


@torch.no_grad()
def run_cell(name: str, args) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = CELLS[name]
    dev = "cuda"
    print(f"[gate] cell={name} model={cfg['model']} ctx={cfg['ctx']} "
          f"K={cfg['K']} target_accept={cfg['e3']}", flush=True)
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], dtype=torch.bfloat16, device_map=dev,
        attn_implementation="sdpa", trust_remote_code=cfg["trust"])
    model.eval()
    n_layers = model.config.num_hidden_layers

    prompts = build_prompts(tok, cfg["ctx"], args.prompts)
    matches, overlaps = [], []
    for pi, ptxt in enumerate(prompts):
        ids = tok(ptxt, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
        T = ids.shape[1]
        # chunked prefill, logits kept only for the last position
        from transformers import DynamicCache
        cache = DynamicCache()
        pos = 0
        logits = None
        while pos < T:
            chunk = ids[:, pos:pos + args.chunk]
            out = model(input_ids=chunk, past_key_values=cache, use_cache=True,
                        logits_to_keep=1)
            cache = out.past_key_values
            logits = out.logits[:, -1]
            pos += chunk.shape[1]
        cur = int(torch.argmax(logits, -1))  # first generated token (target greedy)

        for step in range(args.gen):
            seqlen = T + step          # cache currently holds positions [0, seqlen)
            tok_in = torch.tensor([[cur]], device=dev)
            posid = torch.tensor([[seqlen]], device=dev)

            # ---- DRAFT step: same token, target KV sliced to sinks+window
            dcache = sliced_cache(cache, n_layers, seqlen, args.sinks, args.window)
            dlen = args.sinks + min(args.window, seqlen - args.sinks)
            dout = model(input_ids=tok_in, past_key_values=dcache, use_cache=True,
                         position_ids=posid,
                         cache_position=torch.tensor([dlen], device=dev))
            dlog = dout.logits[:, -1].float()
            del dcache, dout

            # ---- TARGET step (advances the real cache)
            tout = model(input_ids=tok_in, past_key_values=cache, use_cache=True,
                         position_ids=posid,
                         cache_position=torch.tensor([seqlen], device=dev))
            cache = tout.past_key_values
            tlog = tout.logits[:, -1].float()

            t_next = int(torch.argmax(tlog, -1))
            d_next = int(torch.argmax(dlog, -1))
            matches.append(1.0 if d_next == t_next else 0.0)
            overlaps.append(float(torch.minimum(
                torch.softmax(tlog, -1), torch.softmax(dlog, -1)).sum()))
            cur = t_next
        print(f"[gate]  prompt {pi}: running beta_greedy="
              f"{sum(matches)/len(matches):.4f} overlap_T1="
              f"{sum(overlaps)/len(overlaps):.4f}", flush=True)
        del cache
        torch.cuda.empty_cache()

    beta = sum(matches) / len(matches)
    ov = sum(overlaps) / len(overlaps)
    K = cfg["K"]
    tau = (K + 1.0) if beta > 0.9999 else (1 - beta ** (K + 1)) / (1 - beta)
    err = (tau - cfg["e3"]) / cfg["e3"]
    verdict = "PASS" if abs(err) <= 0.03 else ("SOFT" if abs(err) <= 0.06 else "FAIL")
    res = dict(cell=name, beta_greedy=round(beta, 4), overlap_t1=round(ov, 4),
               tau_geom=round(tau, 3), e3_accept=cfg["e3"],
               err_pct=round(100 * err, 1), verdict=verdict,
               positions=len(matches), prompts=args.prompts, gen=args.gen,
               window=args.window, sinks=args.sinks)
    print(f"[gate] RESULT {json.dumps(res)}", flush=True)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", choices=["dense", "moe", "all"], default="all")
    ap.add_argument("--prompts", type=int, default=4)
    ap.add_argument("--gen", type=int, default=96)
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--sinks", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=4096)
    args = ap.parse_args()
    os.environ.setdefault("HF_HOME", "/data/smcho/huggingface")

    cells = ["dense", "moe"] if args.cell == "all" else [args.cell]
    out = [run_cell(c, args) for c in cells]
    (PHASE / "data").mkdir(exist_ok=True)
    f = PHASE / "data/anchor_gate.json"
    f.write_text(json.dumps(out, indent=1))
    print(f"[gate] wrote {f}")
    worst = max(abs(r["err_pct"]) for r in out)
    print(f"[gate] GATE {'OPEN' if all(r['verdict'] != 'FAIL' for r in out) else 'CLOSED'}"
          f" (worst |err| {worst:.1f}%)")
    return 0 if all(r["verdict"] != "FAIL" for r in out) else 1


if __name__ == "__main__":
    sys.exit(main())
