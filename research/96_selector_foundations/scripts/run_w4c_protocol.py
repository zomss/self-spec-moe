#!/usr/bin/env python3
"""W4c: compile-protocol vs serving-protocol R, inside ONE boot.

W4b found the same nominal cell (llama b8/c14k K4 w2048 s-b2, notune)
reports R ~ 0.55 under the compile protocol but realizes R ~ 0.33-0.43 in
serving. This script walks the ladder between the two protocols with the
same engine, same kernels, same boot:

  content x style, per boot (off / spec):
    compile-N160   T(1+N)-T(1), N=160 (the map's EXACT arithmetic,
                   min-of-3 each side, ignore_eos, greedy)
    compile-N512   same arithmetic, N=512 (amortization rung)
    serving        max_tokens=512, natural EOS, wall + engine phase
                   timers (W2 style), 4 rounds
    serving-ie     serving with ignore_eos=True (EOS-stagger rung)
  contents:
    cmap   93's cached compile prompts (byte-identical to the W4b audit)
    r5     load_regime R5 seed 0 (the deployment content)

R per (content, style) = (tau/S - 1)/K with S the spec/off rate ratio of
the SAME (content, style).

Pre-registered predictions:
  P-W4c1  the PROTOCOL axis dominates: R(serving) < R(compile) on both
          contents; the content axis moves R by < 10%.
  P-W4c2  amortization is the mechanism: R(compile-N512) < R(compile-N160),
          moving toward R(serving). If R is N-flat, the mechanism is
          scheduler/batching dynamics, not per-chain startup.
  P-W4c3  EOS stagger is NOT the mechanism: |R(serving) - R(serving-ie)|
          < 5% relative.
  P-W4c4  certification: in-boot serving R5 rates reproduce the W2/W2d
          anchors (off ~142, spec ~195) within 2%.

env: W4C_ARM off|spec   W4C_OUT path
"""
import json
import os
import re
import sys
import time
from pathlib import Path

PHASE = Path(__file__).resolve().parents[1]
P93 = PHASE.parent / "93_c1_grid"
sys.path.insert(0, str(PHASE.parent / "88_regime_eval/scripts"))
from regime_datasets import load_regime  # noqa: E402

ARM = os.environ["W4C_ARM"]           # off | spec
MODEL = "NousResearch/Meta-Llama-3.1-8B-Instruct"
DRAFT = os.path.expanduser("/data/smcho/ckpts/Llama31-8B-Instruct-W4A16-INT4-sym")
B = 8
CTX = 14000
K = 4

HISTS = ["vllm:request_prefill_time_seconds",
         "vllm:request_decode_time_seconds"]


def counters(llm):
    acc = dr = 0
    for m in llm.get_metrics():
        if m.name == "vllm:spec_decode_num_accepted_tokens":
            acc = m.value
        elif m.name == "vllm:spec_decode_num_drafts":
            dr = m.value
    return acc, dr


def hist_snap(llm):
    s = {}
    for m in llm.get_metrics():
        if m.name in HISTS:
            s[m.name] = (m.sum, m.count)
    return s


def compile_style(llm, ps, sp1_kw, n, reps=3):
    """The map's exact arithmetic: min(T(1+N)) - min(T(1))."""
    from vllm import SamplingParams
    sp1 = SamplingParams(max_tokens=1, ignore_eos=True, temperature=0)
    spN = SamplingParams(max_tokens=1 + n, ignore_eos=True, temperature=0)
    llm.generate(ps, spN, use_tqdm=False)          # warm shapes
    t1s, tns, accs = [], [], []
    for _ in range(reps):
        t0 = time.perf_counter()
        llm.generate(ps, sp1, use_tqdm=False)
        t1s.append(time.perf_counter() - t0)
        a0, d0 = counters(llm)
        t0 = time.perf_counter()
        llm.generate(ps, spN, use_tqdm=False)
        tns.append(time.perf_counter() - t0)
        a1, d1 = counters(llm)
        if d1 > d0:
            accs.append(1 + (a1 - a0) / (d1 - d0))
    dec = min(tns) - min(t1s)
    return {"style": f"compile-N{n}", "rate": round(len(ps) * n / dec, 1),
            "accept": round(sum(accs) / len(accs), 3) if accs else None,
            "t1s": [round(t, 3) for t in t1s],
            "tns": [round(t, 3) for t in tns]}


def serving_style(llm, ps, ignore_eos, rounds=4):
    from vllm import SamplingParams
    sp = SamplingParams(max_tokens=512, ignore_eos=ignore_eos,
                        temperature=0)
    llm.generate(ps, sp, use_tqdm=False)           # warm
    out = []
    for _ in range(rounds):
        h0 = hist_snap(llm)
        a0, d0 = counters(llm)
        t0 = time.perf_counter()
        o = llm.generate(ps, sp, use_tqdm=False)
        dt = time.perf_counter() - t0
        h1 = hist_snap(llm)
        a1, d1 = counters(llm)
        ntok = sum(len(x.outputs[0].token_ids) for x in o)
        dec_s = (h1["vllm:request_decode_time_seconds"][0]
                 - h0["vllm:request_decode_time_seconds"][0])
        n_req = (h1["vllm:request_decode_time_seconds"][1]
                 - h0["vllm:request_decode_time_seconds"][1])
        out.append({
            "wall_s": round(dt, 3), "ntok": ntok,
            "e2e_rate": round(ntok / dt, 1),
            "dec_rate_req": round((ntok - n_req) / dec_s, 1)
            if dec_s > 0 else None,
            "accept": round(1 + (a1 - a0) / (d1 - d0), 3)
            if d1 > d0 else None})
    return {"style": "serving-ie" if ignore_eos else "serving",
            "rounds": out}


def main():
    from transformers import AutoTokenizer

    from vllm import LLM

    spec = None
    if ARM == "spec":
        os.environ["VLLM_SELF_SPEC_DRAFT_KV_WINDOW"] = "2048"
        os.environ["VLLM_SELF_SPEC_DRAFT_KV_SINKS"] = "16"
        os.environ["VLLM_SELF_SPEC_DRAFT_SKIP_LAYERS"] = "3,8"
        spec = {"method": "draft_model", "model": DRAFT,
                "num_speculative_tokens": K,
                "draft_tensor_parallel_size": 1}

    llm = LLM(model=MODEL, speculative_config=spec,
              tensor_parallel_size=1, max_model_len=20480,
              gpu_memory_utilization=0.90, max_num_seqs=32,
              enable_prefix_caching=False, disable_log_stats=False,
              async_scheduling=True, max_num_batched_tokens=8192,
              kernel_config={"enable_flashinfer_autotune": False})
    tok = AutoTokenizer.from_pretrained(MODEL)

    # content 1: the audit's exact compile prompts (93 disk cache)
    slug = re.sub(r"[^A-Za-z0-9]+", "_", MODEL)
    cache = P93 / "data" / f"prompts_cache_{slug}_{CTX}.json"
    docs = json.loads(cache.read_text())
    cmap = [docs[i % len(docs)] for i in range(B)]
    # content 2: the deployment's R5 prompts (seed 0)
    r5, _ = load_regime("R5", tok, n=16, seed=0)
    r5 = r5[:B]

    out = {"arm": ARM, "batch": B, "ctx": CTX, "k": K if spec else 0,
           "prompt_cache": str(cache), "contents": {}}
    for cname, ps in (("cmap", cmap), ("r5", r5)):
        res = []
        res.append(compile_style(llm, ps, None, 160))
        res.append(compile_style(llm, ps, None, 512))
        res.append(serving_style(llm, ps, ignore_eos=False))
        res.append(serving_style(llm, ps, ignore_eos=True))
        out["contents"][cname] = res
        for r in res:
            tag = r["style"]
            if "rate" in r:
                print(f"[W4c] {ARM} {cname} {tag:12s} rate={r['rate']} "
                      f"accept={r['accept']}", flush=True)
            else:
                rr = sorted(x["e2e_rate"] for x in r["rounds"])
                print(f"[W4c] {ARM} {cname} {tag:12s} "
                      f"e2e_med={rr[len(rr) // 2]} "
                      f"dec_req={[x['dec_rate_req'] for x in r['rounds']]} "
                      f"accepts={[x['accept'] for x in r['rounds']]}",
                      flush=True)

    path = os.environ.get("W4C_OUT", str(
        PHASE / "data" / "w4" / f"w4c_{ARM}.json"))
    Path(path).write_text(json.dumps(out, indent=1))
    print("[W4c] saved ->", path, flush=True)


if __name__ == "__main__":
    main()
