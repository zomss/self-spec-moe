#!/usr/bin/env python3
"""Real full-EP decode step time vs injected per-collective exposed-A2A delay.

Single-node emulation of inter-node all-to-all cost. Each MoE layer's AgRs
dispatch/combine is delayed by `--delay-us` on the GPU stream (captured into the
CUDA graph via the Phase 02 hook), emulating the latency those collectives would
pay if the EP group spanned nodes over IB.

Because the naive AgRs path serializes the injected _sleep after each collective
(no DBO), the delay is FULLY EXPOSED here -- so this measures the optimistic
(upper-bound) exposed-A2A fraction f. A real DeepEP+DBO system would hide some of
it; that residual is the one quantity that still needs real multi-node hardware.

Per (batch B, delay d): S(B,d) = per-decode-step time. Fit across d gives the
number of injected collectives per step (slope) and the base step time
(intercept); f(B,d) = injected / S(B,d).

Set the delay env BEFORE importing vllm so the worker processes inherit it and
the all2all manager calibrates the GPU clock at init.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np


def parse_int_list(raw: str) -> list[int]:
    return [int(v) for v in raw.split(",") if v.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--data-parallel-size", type=int, default=4)
    ap.add_argument("--batch-sizes", type=parse_int_list, default=[1, 8, 32, 64])
    ap.add_argument("--delay-us", type=float, required=True)
    ap.add_argument("--input-len", type=int, default=4)
    ap.add_argument("--output-len", type=int, default=64)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--output-json", type=Path, required=True)
    a = ap.parse_args()

    # Must be set before importing vllm so workers inherit it.
    os.environ["VLLM_SELF_SPEC_EMULATE_A2A_DELAY_US"] = str(a.delay_us)
    os.environ["VLLM_SELF_SPEC_LOG_A2A_COUNTS"] = "1"

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=a.model,
        tensor_parallel_size=a.tensor_parallel_size,
        data_parallel_size=a.data_parallel_size,
        enable_expert_parallel=True,
        load_format="dummy",
        enforce_eager=False,
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
        max_model_len=2048,
    )
    sp = SamplingParams(
        temperature=1.0, top_p=1.0, ignore_eos=True,
        max_tokens=a.output_len, detokenize=False,
    )

    rows = []
    rng = np.random.default_rng(0)
    for B in a.batch_sizes:
        prompts = [
            {"prompt_token_ids": rng.integers(10000, size=a.input_len).tolist()}
            for _ in range(B)
        ]

        def once() -> float:
            t = time.perf_counter()
            llm.generate(prompts, sampling_params=sp, use_tqdm=False)
            return time.perf_counter() - t

        for _ in range(a.warmup):
            once()
        lat = [once() for _ in range(a.iters)]
        total_ms = float(np.median(lat)) * 1000.0
        step_ms = total_ms / a.output_len  # ms per decode step (emits B tokens)
        rows.append({
            "batch": B,
            "step_ms": round(step_ms, 4),
            "throughput_tok_s": round(B / step_ms * 1000.0, 1),
        })
        print(f"[d={a.delay_us:.0f}us] B={B:>3}: step={step_ms:8.3f} ms  "
              f"thr={B / step_ms * 1000.0:8.1f} tok/s")

    a.output_json.parent.mkdir(parents=True, exist_ok=True)
    a.output_json.write_text(json.dumps({
        "model": a.model,
        "tensor_parallel_size": a.tensor_parallel_size,
        "data_parallel_size": a.data_parallel_size,
        "enable_expert_parallel": True,
        "delay_us": a.delay_us,
        "input_len": a.input_len,
        "output_len": a.output_len,
        "rows": rows,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
