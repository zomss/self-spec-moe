#!/usr/bin/env python3
"""Phase 77 sweep scorer: offline teacher-forced beta per (lever x strength).

Generalizes the anchor-gate primitive (GATE OPEN: dense -1.6%, MoE +0.3% vs
e2e accept lengths) from {window} to every lever. Two stages:

  Stage A (once per model x ctx): target prefill + greedy gen over W7-faithful
    prompts; saves token path, target softmax at gen positions, and the full
    target KV cache -> data/refs/. Every arm then scores against the SAME
    references (paired comparison; requirements from the gate: >=12 prompts x
    96 positions, aggregate-only stats).
  Stage B (per arm): draft decode step at every generated position against the
    TARGET KV (SHARED_KV semantics, as e2e). Per-position caches are built
    from zero-copy views cache[:, :, :t]; the forward's own kv-append is the
    only copy. Records greedy top-1 match + T=1 overlap sum min(p_d, p_t).

Arms (drafts differ ONLY in the delta named):
  win<W>       sinks16 + last-W window slice of the target KV      (gate-validated)
  skip<PCT>    first keep=round(L*(1-pct)) layers + final norm/head (early exit)
  kvq_fp8      draft reads the prefix KV cast e4m3->bf16 (scale 1.0, as P74)
               == the hypothetical DRAFT-ONLY fp8 KV pool
  lr<PCT>      MoE only: router logits masked to the contiguous expert shard
               [0 : E*pct) (rank-0 EP shard semantics, P24/25), renorm natural
  q_int4 / q_fp8   weights RTN fake-quantized IN PLACE (P22 method: int4 g128
               sym / fp8 e4m3 per-channel) -- destructive, so quant arms run
               LAST and each reloads the model fresh.

Usage:
  python score_accept.py --model dense --ctx 16384 [--arms win512,skip50,...]
Appends rows to data/beta.csv.
"""

import argparse
import csv
import gc
import json
import sys
from pathlib import Path

import torch

PHASE = Path(__file__).resolve().parent.parent
REPO = PHASE.parent.parent
sys.path.insert(0, str(PHASE / "scripts"))
from anchor_gate import FILLER, PROMPTS, build_prompts, cache_kv  # noqa: E402

MODELS = {
    "dense": dict(hf="Qwen/Qwen2.5-7B-Instruct", trust=False, moe=False, layers=28),
    "moe": dict(hf="Qwen/Qwen3-30B-A3B", trust=False, moe=True, layers=48),
    "mla": dict(hf="deepseek-ai/DeepSeek-V2-Lite", trust=True, moe=True, layers=27),
}
DEFAULT_ARMS = {
    False: "win128,win512,win2048,skip125,skip25,skip375,skip50,kvq_fp8,q_int4,q_fp8",
    True: "win128,win512,win2048,skip125,skip25,skip375,skip50,kvq_fp8,lr25,lr50,q_int4,q_fp8",
}
SINKS = 16


# ------------------------------------------------------------------ stage A
@torch.no_grad()
def stage_a(model, tok, cfg, args, refdir: Path):
    from transformers import DynamicCache
    prompts = build_prompts(tok, args.ctx, args.prompts)
    refdir.mkdir(parents=True, exist_ok=True)
    for pi, ptxt in enumerate(prompts):
        if (refdir / f"p{pi}_meta.pt").exists():
            continue
        ids = tok(ptxt, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
        T = ids.shape[1]
        cache = DynamicCache()
        pos, logits = 0, None
        while pos < T:
            out = model(input_ids=ids[:, pos:pos + args.chunk], past_key_values=cache,
                        use_cache=True, logits_to_keep=1)
            cache, logits = out.past_key_values, out.logits[:, -1]
            pos += min(args.chunk, T - pos)
        toks, probs = [], []
        cur = int(torch.argmax(logits, -1))
        for step in range(args.gen):
            toks.append(cur)
            seqlen = T + step
            out = model(input_ids=torch.tensor([[cur]], device="cuda"),
                        past_key_values=cache, use_cache=True,
                        position_ids=torch.tensor([[seqlen]], device="cuda"),
                        cache_position=torch.tensor([seqlen], device="cuda"))
            cache = out.past_key_values
            p = torch.softmax(out.logits[:, -1].float(), -1)
            probs.append(p.half().cpu())
            cur = int(torch.argmax(p, -1))
        n_layers = len(getattr(cache, "layers", None) or cache.key_cache)
        kv = {li: tuple(x.cpu() for x in cache_kv(cache, li)) for li in range(n_layers)}
        torch.save(dict(prompt_ids=ids.cpu(), gen_tokens=toks,
                        probs=torch.cat(probs)), refdir / f"p{pi}_meta.pt")
        torch.save(kv, refdir / f"p{pi}_cache.pt")
        print(f"[A] ref p{pi}: T={T} gen={args.gen}", flush=True)
        del cache
        torch.cuda.empty_cache()


# ------------------------------------------------------- draft-side machinery
def view_cache(kv_gpu, n_layers, upto, window=None, sinks=SINKS):
    """DynamicCache over [0:upto) views; optional sinks+window slice."""
    from transformers import DynamicCache
    c = DynamicCache()
    for li in range(n_layers):
        k, v = kv_gpu[li]
        if window is not None and upto > sinks + window:
            ks = torch.cat([k[:, :, :sinks], k[:, :, upto - window:upto]], dim=2)
            vs = torch.cat([v[:, :, :sinks], v[:, :, upto - window:upto]], dim=2)
        else:
            ks, vs = k[:, :, :upto], v[:, :, :upto]
        c.update(ks, vs, li)
    return c


class LayerSkip:
    """Skip a contiguous MIDDLE block of decoder layers (SWIFT-style,
    training-free). Early exit (keeping only the FIRST layers) was tried and
    is DEAD: beta=0.000, overlap 0.003 -- the mid-stack residual stream is not
    in the lm_head's input space. Keeping the last 2 layers re-normalizes.
    Kept layers retain their original layer_idx, so they index the target
    cache correctly."""
    KEEP_TAIL = 2

    def __init__(self, model, n_layers: int, frac: float):
        self.m = model.model
        self.cfg = model.config
        n_skip = round(n_layers * frac)
        a = max(1, n_layers - n_skip - self.KEEP_TAIL)
        self.kept = [i for i in range(n_layers) if not (a <= i < a + n_skip)]

    def __enter__(self):
        self.layers, self.n = self.m.layers, self.cfg.num_hidden_layers
        import torch.nn as nn
        self.m.layers = nn.ModuleList([self.layers[i] for i in self.kept])
        self.cfg.num_hidden_layers = len(self.kept)

    def __exit__(self, *a):
        self.m.layers, self.cfg.num_hidden_layers = self.layers, self.n


class RouterMask:
    """Mask MoE routing to the contiguous expert shard [0 : n_keep).

    transformers 5.x routers (Qwen3MoeTopKRouter) do logits->softmax->topk
    INSIDE forward, so the hook RECOMPUTES routing from masked logits --
    exactly P25's semantics: non-resident logits to -inf, topk over survivors,
    renorm per norm_topk_prob. Duck-typed on (top_k, num_experts, weight)."""

    def __init__(self, model, frac: float):
        self.hooks = []
        self.model = model
        self.frac = frac

    def __enter__(self):
        import torch.nn.functional as F

        def mk(router):
            n_keep = max(1, int(router.num_experts * self.frac))

            def hook(mod, inp, out):
                hidden = inp[0].reshape(-1, mod.hidden_dim)
                logits = F.linear(hidden, mod.weight)
                logits[..., n_keep:] = float("-inf")
                probs = torch.nn.functional.softmax(logits, dtype=torch.float, dim=-1)
                val, idx = torch.topk(probs, mod.top_k, dim=-1)
                if mod.norm_topk_prob:
                    val = val / val.sum(-1, keepdim=True)
                return probs, val.to(probs.dtype), idx
            return hook

        for mod in self.model.modules():
            gate = getattr(mod, "gate", None)
            if gate is not None and all(hasattr(gate, a) for a in
                                        ("top_k", "num_experts", "weight")):
                self.hooks.append(gate.register_forward_hook(mk(gate)))
        assert self.hooks, "no MoE router modules found for RouterMask"
        return self

    def __exit__(self, *a):
        for h in self.hooks:
            h.remove()
        self.hooks = []


def fake_quant_(model, kind: str):
    """P22-style RTN in place (skips lm_head/embeddings). DESTRUCTIVE."""
    import torch.nn as nn
    n = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear) or "lm_head" in name:
            continue
        w = mod.weight.data
        if kind == "int4":                       # g128 symmetric
            out_f, in_f = w.shape
            g = 128 if in_f % 128 == 0 else in_f
            wg = w.float().view(out_f, in_f // g, g)
            s = wg.abs().amax(-1, keepdim=True).clamp(min=1e-8) / 7.0
            w.copy_(((wg / s).round().clamp(-8, 7) * s).view(out_f, in_f).to(w.dtype))
        elif kind == "fp8":                      # e4m3 per-out-channel
            s = w.float().abs().amax(-1, keepdim=True).clamp(min=1e-8) / 448.0
            w.copy_(((w.float() / s).to(torch.float8_e4m3fn).float() * s).to(w.dtype))
        n += 1
    print(f"[B] fake-quant {kind}: {n} Linear layers", flush=True)


def kv_quant(kv_gpu):
    """Prefix KV cast e4m3->bf16 (scale 1.0, matching P74's fp8_e4m3 pool)."""
    return {li: tuple(t.clamp(-448, 448).to(torch.float8_e4m3fn).to(t.dtype)
                      for t in kv_gpu[li]) for li in kv_gpu}


# ------------------------------------------------------------------ stage B
@torch.no_grad()
def score_arm(model, cfg, arm: str, refdir: Path, args) -> dict:
    n_layers = model.config.num_hidden_layers
    dev = "cuda"
    matches, overlaps = [], []
    ctxman = None
    window = None
    if arm.startswith("win"):
        window = int(arm[3:])
    elif arm.startswith("skip"):
        pct = int(arm[4:]) / (1000 if len(arm) > 6 else 100)  # skip125 -> 12.5%
        ctxman = LayerSkip(model, cfg["layers"], pct)
    elif arm.startswith("lr"):
        ctxman = RouterMask(model, int(arm[2:]) / 100)
    # kvq_fp8 / q_* need no forward-time context (cache transform / weights)

    for pi in range(args.prompts):
        meta = torch.load(refdir / f"p{pi}_meta.pt", weights_only=False)
        kv_cpu = torch.load(refdir / f"p{pi}_cache.pt", weights_only=False)
        kv = {li: tuple(t.to(dev) for t in kv_cpu[li]) for li in kv_cpu}
        if arm == "kvq_fp8":
            kv = kv_quant(kv)
        T = meta["prompt_ids"].shape[1]
        toks, probs = meta["gen_tokens"], meta["probs"]
        # kept layers retain original layer_idx -> cache always holds ALL layers
        eff_layers = n_layers
        import contextlib
        with (ctxman or contextlib.nullcontext()):
            for step, cur in enumerate(toks):
                seqlen = T + step
                dcache = view_cache(kv, eff_layers, seqlen, window=window)
                dlen = dcache.get_seq_length()
                out = model(input_ids=torch.tensor([[cur]], device=dev),
                            past_key_values=dcache, use_cache=True,
                            position_ids=torch.tensor([[seqlen]], device=dev),
                            cache_position=torch.tensor([dlen], device=dev))
                dp = torch.softmax(out.logits[:, -1].float(), -1)
                tp = probs[step].to(dev).float()
                matches.append(float(int(dp.argmax()) == int(tp.argmax())))
                overlaps.append(float(torch.minimum(dp[0], tp).sum()))
                del dcache, out
        del kv, kv_cpu
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[B] {arm} p{pi}: beta={sum(matches)/len(matches):.4f}", flush=True)

    return dict(model=args.model, ctx=args.ctx, arm=arm,
                beta_greedy=round(sum(matches) / len(matches), 4),
                overlap_t1=round(sum(overlaps) / len(overlaps), 4),
                positions=len(matches), prompts=args.prompts)


def load_model(cfg):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(
        cfg["hf"], dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa", trust_remote_code=cfg["trust"])
    m.eval()
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--ctx", type=int, required=True)
    ap.add_argument("--arms", default=None)
    ap.add_argument("--prompts", type=int, default=12)
    ap.add_argument("--gen", type=int, default=96)
    ap.add_argument("--chunk", type=int, default=4096)
    args = ap.parse_args()
    cfg = MODELS[args.model]
    arms = (args.arms or DEFAULT_ARMS[cfg["moe"]]).split(",")
    # quant arms mutate weights in place -> force them last
    arms = sorted(arms, key=lambda a: a.startswith("q_"))

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["hf"], trust_remote_code=cfg["trust"])
    refdir = PHASE / f"data/refs/{args.model}_c{args.ctx}"
    model = load_model(cfg)
    stage_a(model, tok, cfg, args, refdir)

    out = PHASE / "data/beta.csv"
    done = set()
    if out.exists():  # resume: skip (model, ctx, arm) rows already recorded
        done = {(r["model"], r["ctx"], r["arm"])
                for r in csv.DictReader(out.open())}
    mutated = False
    n_written = 0
    for arm in arms:
        if (args.model, str(args.ctx), arm) in done:
            print(f"[B] skip {arm} (already in beta.csv)", flush=True)
            continue
        if arm.startswith("q_"):
            if mutated:
                del model
                gc.collect(); torch.cuda.empty_cache()
                model = load_model(cfg)
            fake_quant_(model, arm[2:])
            mutated = True
        r = score_arm(model, cfg, arm, refdir, args)
        print(f"[B] RESULT {json.dumps(r)}", flush=True)
        # write IMMEDIATELY: a later arm's crash must not lose earlier arms
        # (the first MoE run lost 3 ctx x 7 arms to the lr25 TypeError)
        exists = out.exists()
        with out.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(r.keys()))
            if not exists:
                w.writeheader()
            w.writerow(r)
        n_written += 1
    print(f"[B] appended {n_written} rows -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
