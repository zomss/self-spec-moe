#!/usr/bin/env python3
"""Single-engine decode latency vs batch for a given TP/EP config.

Used to compare EP=2 (TP2+EP, 2 GPUs) vs EP=4 (TP4+EP, 4 GPUs): how much does the
expert all-to-all cost as the EP group grows? Dummy weights (timing only).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def parse_int_list(raw: str) -> list[int]:
    return [int(v) for v in raw.split(",") if v.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tensor-parallel-size", type=int, required=True)
    ap.add_argument("--enable-expert-parallel", action="store_true")
    ap.add_argument("--batch-sizes", type=parse_int_list, required=True)
    ap.add_argument("--input-len", type=int, default=4)
    ap.add_argument("--output-len", type=int, default=64)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--label", required=True)
    ap.add_argument("--output-json", type=Path, required=True)
    a = ap.parse_args()

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=a.model,
        tensor_parallel_size=a.tensor_parallel_size,
        enable_expert_parallel=a.enable_expert_parallel,
        load_format="dummy",
        enforce_eager=False,
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
        max_model_len=2048,
    )
    sp = SamplingParams(
        temperature=1.0, top_p=1.0, ignore_eos=True, max_tokens=a.output_len, detokenize=False
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
        ms_per_tok = total_ms / a.output_len
        rows.append(
            {
                "batch": B,
                "total_ms_median": round(total_ms, 3),
                "ms_per_token": round(ms_per_tok, 4),
            }
        )
        print(f"[{a.label}] B={B:>3}: {ms_per_tok:8.3f} ms/token  (total {total_ms:.1f} ms)")

    a.output_json.parent.mkdir(parents=True, exist_ok=True)
    a.output_json.write_text(
        json.dumps(
            {
                "label": a.label,
                "model": a.model,
                "tensor_parallel_size": a.tensor_parallel_size,
                "enable_expert_parallel": a.enable_expert_parallel,
                "input_len": a.input_len,
                "output_len": a.output_len,
                "rows": rows,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
