#!/usr/bin/env python3
"""W14 item A — phase-local equal-work measurement runner.

Phase-local by design (w14_plan.md item A): does NOT mutate a prior
phase's runner. Emits RAW quantities only; the scorer derives T, P, q
and their uncertainty.

Per measurement interval it records:
  - vllm:request_decode_time_seconds sum/count deltas (decode currency;
    end-to-end wall clock carries prefill dilution -- W13 measured 8.4%
    mean / 17.7% max apparent transfer error from that alone)
  - output-token, request, accepted-token, draft-token counts
  - K, complete configuration id, realization, graph bucket
  - observed n_active, total-scheduled-KV and generated-suffix ranges

  T = 1 / rate_AR        (AR arm)
  P = tau / rate_spec    (spec arm)
  q = P / T = tau / S_dec

Valid ONLY for equal-work runs over a matched state interval.

env: W14_MODEL W14_DRAFT W14_K W14_WINDOW W14_SKIP W14_REGIMES
     W14_BATCHES W14_FIXED_LEN W14_ITERS W14_SEEDS W14_OUT W14_TAG
     W14_REALIZATION
"""
import json
import os
import sys
import time
from pathlib import Path

PHASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE.parent / "88_regime_eval/scripts"))
from regime_datasets import load_regime  # noqa: E402

MODEL = os.environ.get("W14_MODEL", "Qwen/Qwen3-8B")
DRAFT = os.environ.get("W14_DRAFT", "off")
K = int(os.environ.get("W14_K", "4"))
WINDOW = int(os.environ.get("W14_WINDOW", "0"))
SKIP = os.environ.get("W14_SKIP", "")
REGIMES = os.environ.get("W14_REGIMES", "R5,R5cot").split(",")
BATCHES = [int(b) for b in os.environ.get("W14_BATCHES", "1,8").split(",")]
FIXED_LEN = int(os.environ.get("W14_FIXED_LEN", "512"))
ITERS = int(os.environ.get("W14_ITERS", "4"))
SEEDS = [int(s) for s in os.environ.get("W14_SEEDS", "0,1").split(",")]
TAG = os.environ.get("W14_TAG", "w14")
REALIZATION = os.environ.get("W14_REALIZATION", "piecewise")

DEC_HIST = "vllm:request_decode_time_seconds"
PRE_HIST = "vllm:request_prefill_time_seconds"


def metrics(llm):
    """Raw cumulative counters + histogram (sum, count) pairs."""
    out = {"acc": 0, "drafts": 0, "preempt": 0,
           "dec_sum": 0.0, "dec_n": 0, "pre_sum": 0.0, "pre_n": 0}
    for m in llm.get_metrics():
        if m.name == "vllm:spec_decode_num_accepted_tokens":
            out["acc"] = m.value
        elif m.name == "vllm:spec_decode_num_drafts":
            out["drafts"] = m.value
        elif m.name == "vllm:num_preemptions":
            out["preempt"] = m.value
        elif m.name == DEC_HIST:
            out["dec_sum"], out["dec_n"] = m.sum, m.count
        elif m.name == PRE_HIST:
            out["pre_sum"], out["pre_n"] = m.sum, m.count
    return out


def main():
    from transformers import AutoTokenizer

    from vllm import LLM, SamplingParams

    spec = None
    if DRAFT != "off":
        if WINDOW:
            os.environ["VLLM_SELF_SPEC_DRAFT_KV_WINDOW"] = str(WINDOW)
            os.environ["VLLM_SELF_SPEC_DRAFT_KV_SINKS"] = "16"
        if SKIP:
            os.environ["VLLM_SELF_SPEC_DRAFT_SKIP_LAYERS"] = SKIP
        spec = {"method": "draft_model", "model": os.path.expanduser(DRAFT),
                "num_speculative_tokens": K, "draft_tensor_parallel_size": 1}

    llm = LLM(model=MODEL, speculative_config=spec, tensor_parallel_size=1,
              max_model_len=20480, gpu_memory_utilization=0.90,
              max_num_seqs=max(BATCHES), enable_prefix_caching=False,
              disable_log_stats=False, async_scheduling=True,
              max_num_batched_tokens=8192,
              kernel_config={"enable_flashinfer_autotune": False})
    tok = AutoTokenizer.from_pretrained(MODEL)

    out = {"tag": TAG, "model": MODEL, "draft": DRAFT,
           "K": K if spec else 0, "window": WINDOW, "skip": SKIP,
           "realization": REALIZATION, "fixed_len": FIXED_LEN,
           "iters": ITERS, "seeds": SEEDS, "tune": False,
           "max_num_seqs": max(BATCHES), "cells": []}

    for rid in REGIMES:
        for seed in SEEDS:
            prompts, gs = load_regime(rid, tok, n=max(BATCHES), seed=seed)
            plens = [len(tok.encode(p)) for p in prompts]
            sp = SamplingParams(max_tokens=FIXED_LEN,
                                temperature=gs["temperature"],
                                seed=0 if gs["temperature"] else None,
                                ignore_eos=True)
            llm.generate(prompts[:2], SamplingParams(
                max_tokens=32, temperature=gs["temperature"],
                ignore_eos=True), use_tqdm=False)          # warm
            for b in BATCHES:
                if b > len(prompts):
                    continue
                rounds = []
                for _ in range(ITERS):
                    m0 = metrics(llm)
                    t0 = time.perf_counter()
                    outs = llm.generate(prompts[:b], sp, use_tqdm=False)
                    wall = time.perf_counter() - t0
                    m1 = metrics(llm)
                    toks = [len(o.outputs[0].token_ids) for o in outs]
                    d_sum = m1["dec_sum"] - m0["dec_sum"]
                    d_n = m1["dec_n"] - m0["dec_n"]
                    n_out = sum(toks)
                    n_req = len(outs)
                    d_acc = m1["acc"] - m0["acc"]
                    d_dr = m1["drafts"] - m0["drafts"]
                    rounds.append({
                        # RAW -- the scorer derives everything
                        "wall_s": round(wall, 6),
                        "out_tokens": n_out, "n_requests": n_req,
                        "dec_sum_s": round(d_sum, 6), "dec_count": d_n,
                        "pre_sum_s": round(m1["pre_sum"] - m0["pre_sum"], 6),
                        "pre_count": m1["pre_n"] - m0["pre_n"],
                        "accepted_tokens": d_acc, "drafts": d_dr,
                        "preemptions": m1["preempt"] - m0["preempt"],
                        "out_tok_min": min(toks), "out_tok_max": max(toks),
                        # decode-currency per-request rate: the first token
                        # of each request comes from prefill, so the decode
                        # numerator is (out_tokens - n_requests)
                        "dec_rate_req": round((n_out - n_req) / d_sum, 4)
                        if d_sum > 0 else None,
                        "tau": round(1 + d_acc / d_dr, 4) if d_dr > 0 else None,
                    })
                cell = {"rid": rid, "seed": seed, "n_active": b,
                        "prompt_tok_p50": sorted(plens)[len(plens) // 2],
                        "total_kv_lo": b * (sorted(plens)[len(plens) // 2]),
                        "total_kv_hi": b * (sorted(plens)[len(plens) // 2]
                                            + FIXED_LEN),
                        "gen_interval": [0, FIXED_LEN],
                        "graph_bucket": b, "temp": gs["temperature"],
                        "rounds": rounds}
                out["cells"].append(cell)
                print(f"[W14] {rid} s{seed} b={b} "
                      f"dec_rate={[r['dec_rate_req'] for r in rounds]} "
                      f"tau={[r['tau'] for r in rounds]} "
                      f"outtok={rounds[0]['out_tok_min']}-"
                      f"{rounds[0]['out_tok_max']}", flush=True)
        p = os.environ.get("W14_OUT", str(PHASE / "data/w14" / f"{TAG}.json"))
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        Path(p).write_text(json.dumps(out, indent=1))
    out["complete"] = True
    p = os.environ.get("W14_OUT", str(PHASE / "data/w14" / f"{TAG}.json"))
    Path(p).write_text(json.dumps(out, indent=1))
    print("[W14] saved ->", p, flush=True)


if __name__ == "__main__":
    main()
