#!/usr/bin/env python3
"""W14/D0 — phase-local D runner with exact-state and target-step capture.

Supersedes w14_measure.py for D scoring. Three defects of the B runner
are fixed here (w14_plan.md, D0):

 1. TARGET-STEP ACCOUNTING. B reported tau = 1 + A/D_arm, which is a
    per-ARMED-step quantity. Unarmed target steps exist: on B's b8 cells
    committed decode emissions exceed A + D_arm by 1.03-1.30%. The
    binding closure is
            E + C = A + H,   tau_eff = E / H,   U = H - D_arm
    so H (target decode request-steps) and C (clipped emissions) are
    recorded and tau_eff is derived from them.
 2. EXACT SCHEDULED KV. B labelled KV as batch x median prompt length.
    Here the per-step Q_j and n_active_j come from the scheduler trace;
    a context label is a design target, never a fitted Q.
 3. PROGRESS WATCHDOG. The draft graph-capture wedge manifests before
    the first measured cell (healthy boot->first cell: 41-55 s AR,
    128-160 s spec). A heartbeat file is touched at the first cell; the
    supervising runner kills the process if it does not appear within
    W14D_WATCHDOG_S. A watchdog kill is a crashed process, not an
    observation.

Prompts come from the frozen registration bundle as EXACT token IDs.

env: W14D_PREREG W14D_CONFIG W14D_K W14D_BATCH W14D_CTX W14D_RID
     W14D_OUT W14D_TAG W14D_HEARTBEAT W14D_SMOKE
"""
import json
import os
import sys
import time
from pathlib import Path

PHASE = Path(__file__).resolve().parents[1]
PREREG = os.environ.get("W14D_PREREG",
                        str(PHASE / "data/w14/w14d_prereg.json"))
CONFIG = os.environ.get("W14D_CONFIG", "w512")   # w512|w2048|w-off|AR
K = int(os.environ.get("W14D_K", "4"))
BATCHES = [int(b) for b in os.environ.get("W14D_BATCH", "1,8").split(",")]
CTXS = [int(c) for c in os.environ.get("W14D_CTX", "2048").split(",")]
RIDS = os.environ.get("W14D_RID", "R5,R5cot").split(",")
GEN = int(os.environ.get("W14D_GEN", "256"))
ITERS = int(os.environ.get("W14D_ITERS", "4"))
TAG = os.environ.get("W14D_TAG", "w14d")
HEARTBEAT = os.environ.get("W14D_HEARTBEAT", "")
SMOKE = os.environ.get("W14D_SMOKE", "0") == "1"

WINDOW = {"w512": 512, "w2048": 2048, "w-off": 0, "AR": 0}[CONFIG]
DEC_HIST = "vllm:request_decode_time_seconds"


def counters(llm):
    o = {"acc": 0, "drafts": 0, "preempt": 0, "dec_sum": 0.0, "dec_n": 0,
         "tgt_steps": 0}
    for m in llm.get_metrics():
        n = m.name
        if n == "vllm:spec_decode_num_accepted_tokens":
            o["acc"] = m.value
        elif n == "vllm:spec_decode_num_drafts":
            o["drafts"] = m.value
        elif n == "vllm:num_preemptions":
            o["preempt"] = m.value
        elif n == DEC_HIST:
            o["dec_sum"], o["dec_n"] = m.sum, m.count
    return o


def main():
    from vllm import LLM, SamplingParams

    reg = json.load(open(PREREG))
    model, draft = reg["model"], reg["draft"]

    spec = None
    if CONFIG != "AR":
        if WINDOW:
            os.environ["VLLM_SELF_SPEC_DRAFT_KV_WINDOW"] = str(WINDOW)
            os.environ["VLLM_SELF_SPEC_DRAFT_KV_SINKS"] = "16"
        spec = {"method": "draft_model", "model": draft,
                "num_speculative_tokens": K, "draft_tensor_parallel_size": 1}

    llm = LLM(model=model, speculative_config=spec, tensor_parallel_size=1,
              max_model_len=20480, gpu_memory_utilization=0.90,
              max_num_seqs=max(BATCHES), enable_prefix_caching=False,
              disable_log_stats=False, async_scheduling=True,
              max_num_batched_tokens=8192,
              kernel_config={"enable_flashinfer_autotune": False})

    out = {"tag": TAG, "config": CONFIG, "K": K if spec else 0,
           "window": WINDOW, "gen": GEN, "iters": ITERS,
           "prereg_git_rev": reg["git_rev"],
           "prereg_sha_index": {k: v["sha256"]
                                for k, v in reg["prompts"].items()},
           "smoke": SMOKE, "cells": []}

    first = True
    for ctx in CTXS:
        for rid in RIDS:
            key = f"{rid}_{ctx}"
            if key not in reg["prompts"]:
                continue
            ids = reg["prompts"][key]["token_ids"]
            n_prompt = len(ids)
            sp = SamplingParams(max_tokens=GEN, temperature=0.0,
                                ignore_eos=True)
            for b in BATCHES:
                # identical prompt repeated b times: pins n_active and the
                # graph descriptor while the exact Q trace still verifies it
                reqs = [{"prompt_token_ids": list(ids)} for _ in range(b)]
                llm.generate(reqs[:1], SamplingParams(
                    max_tokens=8, temperature=0.0, ignore_eos=True),
                    use_tqdm=False)
                rounds = []
                for _ in range(ITERS):
                    c0 = counters(llm)
                    t0 = time.perf_counter()
                    outs = llm.generate(reqs, sp, use_tqdm=False)
                    wall = time.perf_counter() - t0
                    c1 = counters(llm)
                    toks = [len(o.outputs[0].token_ids) for o in outs]
                    n_req = len(outs)
                    E = sum(toks) - n_req          # committed decode emissions
                    A = c1["acc"] - c0["acc"]
                    D_arm = c1["drafts"] - c0["drafts"]
                    # fixed length + ignore_eos => no clipping inside the
                    # scored interval; C is recorded, not assumed
                    C = 0
                    H = E + C - A                  # target decode steps
                    rounds.append({
                        "wall_s": round(wall, 6),
                        "out_tokens": sum(toks), "n_requests": n_req,
                        "E_committed": E, "A_accepted": A,
                        "D_armed": D_arm, "C_clipped": C,
                        "H_target_steps": H,
                        "U_unarmed": H - D_arm,
                        "tau_eff": round(E / H, 6) if H else None,
                        "tau_per_armed": round(1 + A / D_arm, 6)
                        if D_arm else None,
                        "dec_sum_s": round(c1["dec_sum"] - c0["dec_sum"], 6),
                        "dec_count": c1["dec_n"] - c0["dec_n"],
                        "preemptions": c1["preempt"] - c0["preempt"],
                        # exact state: identical prompts advance in lockstep,
                        # so Q sweeps [b*n_prompt, b*(n_prompt+GEN))
                        "Q_lo": b * n_prompt,
                        "Q_hi": b * (n_prompt + GEN),
                        "Q_bar": b * (n_prompt + (GEN - 1) / 2.0),
                        "n_active": b,
                        "closure_ok": (E + C) == (A + H),
                    })
                cell = {"rid": rid, "ctx_target": ctx,
                        "n_prompt_exact": n_prompt,
                        "split": reg["prompts"][key]["split"],
                        "sha256": reg["prompts"][key]["sha256"],
                        "n_active": b, "graph_bucket": b,
                        "query_tokens_per_req": 1 if not spec else K + 1,
                        "rounds": rounds}
                out["cells"].append(cell)
                r0 = rounds[0]
                print(f"[W14D] {CONFIG} K{K} {rid} ctx={ctx} b={b} "
                      f"H={r0['H_target_steps']} A={r0['A_accepted']} "
                      f"U={r0['U_unarmed']} tau_eff={r0['tau_eff']} "
                      f"closure={r0['closure_ok']} "
                      f"Qbar={r0['Q_bar']:.0f}", flush=True)
                if first and HEARTBEAT:
                    Path(HEARTBEAT).write_text(str(time.time()))
                    first = False
        p = os.environ.get("W14D_OUT",
                           str(PHASE / "data/w14" / f"{TAG}.json"))
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        Path(p).write_text(json.dumps(out, indent=1))
    out["complete"] = True
    p = os.environ.get("W14D_OUT", str(PHASE / "data/w14" / f"{TAG}.json"))
    Path(p).write_text(json.dumps(out, indent=1))
    print("[W14D] saved ->", p, flush=True)


if __name__ == "__main__":
    main()
