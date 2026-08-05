#!/usr/bin/env python3
"""W6 item 2: dense Round-2 real-acceptance calibration (w6_design.md §2).

One boot = one (skip, window) of the q-hum pool (or the AR anchor).
UNCONDITIONAL K4 spec (no policy file) -> pure-K accept, f = (tau-1)/4;
K2's S composes from the same f (tau = 1 + f*K, C2 doctrine). Regimes
R5, R5cot, R1 (the deciding cells), seeds 0 AND 1 (real content draws,
G6). Serving-faithful protocol: same SamplingParams conventions as W2.
Records accept + e2e + per-request decode rate per round (timers free).

Pre-registered:
  P-W6a  measured f shrinks the deciding intervals below the W3 margins
         (the tie-sets become rankable or provably tied)
  P-W6b  seed-0 vs seed-1 f agree within the 1% noise floor + content
         band (content draws are stable at n=16-prompt scale)

env: W6_SKIP 2,8|none   W6_WINDOW 512|2048|off   W6_OUT
"""
import json
import os
import sys
import time
from pathlib import Path

PHASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE.parent / "88_regime_eval/scripts"))
from regime_datasets import load_regime  # noqa: E402

SKIP = os.environ.get("W6_SKIP", "2,8")
WINDOW = os.environ.get("W6_WINDOW", "512")
MODEL = "Qwen/Qwen3-8B"
DRAFT = os.path.expanduser("~/ckpts/Qwen3-8B-W4A8-gptq")
REGIMES = os.environ.get("W6_REGIMES", "R5,R5cot,R1").split(",")
SEEDS = [0, 1]
ITERS = 4

HISTS = ["vllm:request_decode_time_seconds"]


def counters(llm):
    acc = dr = 0
    for m in llm.get_metrics():
        if m.name == "vllm:spec_decode_num_accepted_tokens":
            acc = m.value
        elif m.name == "vllm:spec_decode_num_drafts":
            dr = m.value
    return acc, dr


def dec_snap(llm):
    for m in llm.get_metrics():
        if m.name in HISTS:
            return (m.sum, m.count)
    return (0.0, 0)


def main():
    from transformers import AutoTokenizer

    from vllm import LLM, SamplingParams

    spec = None
    if WINDOW != "off":
        os.environ["VLLM_SELF_SPEC_DRAFT_KV_WINDOW"] = WINDOW
        os.environ["VLLM_SELF_SPEC_DRAFT_KV_SINKS"] = "16"
        if SKIP != "none":
            os.environ["VLLM_SELF_SPEC_DRAFT_SKIP_LAYERS"] = SKIP
        spec = {"method": "draft_model", "model": DRAFT,
                "num_speculative_tokens": 4,
                "draft_tensor_parallel_size": 1}

    llm = LLM(model=MODEL, speculative_config=spec,
              tensor_parallel_size=1, max_model_len=20480,
              gpu_memory_utilization=0.90, max_num_seqs=32,
              enable_prefix_caching=False, disable_log_stats=False,
              async_scheduling=True, max_num_batched_tokens=8192,
              kernel_config={"enable_flashinfer_autotune": False})
    tok = AutoTokenizer.from_pretrained(MODEL)

    out = {"arch": "dense", "skip": SKIP if spec else None,
           "window": WINDOW, "uncond_k": 4 if spec else 0,
           "draft": DRAFT if spec else None, "cells": {}}
    for rid in REGIMES:
        for seed in SEEDS:
            prompts, gs = load_regime(rid, tok, n=16, seed=seed)
            b = gs["batch"]
            sp = SamplingParams(max_tokens=gs["max_tokens"],
                                temperature=gs["temperature"],
                                ignore_eos=(rid == "R5cot"))
            llm.generate(prompts[:b], sp, use_tqdm=False)   # warm
            rounds = []
            for _ in range(ITERS):
                d0 = dec_snap(llm)
                a0, dr0 = counters(llm)
                t0 = time.perf_counter()
                ntok = 0
                if b == 1:
                    for p in prompts[:4]:
                        o = llm.generate([p], sp, use_tqdm=False)
                        ntok += sum(len(x.outputs[0].token_ids)
                                    for x in o)
                else:
                    o = llm.generate(prompts[:b], sp, use_tqdm=False)
                    ntok = sum(len(x.outputs[0].token_ids) for x in o)
                dt = time.perf_counter() - t0
                d1 = dec_snap(llm)
                a1, dr1 = counters(llm)
                dec_s, n_req = d1[0] - d0[0], d1[1] - d0[1]
                rounds.append({
                    "e2e_rate": round(ntok / dt, 1), "ntok": ntok,
                    "dec_rate_req": round((ntok - n_req) / dec_s, 1)
                    if dec_s > 0 else None,
                    "accept": round(1 + (a1 - a0) / (dr1 - dr0), 4)
                    if dr1 > dr0 else None,
                    "drafts": dr1 - dr0})
            out["cells"][f"{rid}_s{seed}"] = rounds
            accs = [r["accept"] for r in rounds if r["accept"]]
            print(f"[W6cal] skip={SKIP} w={WINDOW} {rid} seed={seed} "
                  f"b={b} accepts={accs} "
                  f"e2e={[r['e2e_rate'] for r in rounds]}", flush=True)

    path = os.environ.get("W6_OUT", str(
        PHASE / "data" / "w6" /
        f"w6cal_dense_s-{SKIP.replace(',', '_')}_w{WINDOW}.json"))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(out, indent=1))
    print("[W6cal] saved ->", path, flush=True)


if __name__ == "__main__":
    main()
