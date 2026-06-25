#!/usr/bin/env python3
"""Attention-DP + EP decode step time vs injected per-collective A2A delay.

This exercises the AgRsAll2AllManager dispatch/combine path (the hooked one),
which only fires with data_parallel_size > 1 -- the DeepSeek-style attention-DP +
expert-EP layout that real multi-node MoE decode uses. Each DP rank is a separate
process (per the supported offline-DP pattern); the MoE all_gatherv/reduce_scatterv
synchronizes them every layer, and `--delay-us` is injected on each such collective
on the GPU stream (captured into the CUDA graph).

Per-rank batch B means global batch B*dp at the expert all-gather. Rank 0 writes
the step-time result. Set the delay env before spawning so children inherit it.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Process
from pathlib import Path

import numpy as np


def parse_int_list(raw: str) -> list[int]:
    return [int(v) for v in raw.split(",") if v.strip()]


def worker(rank, dp, tp, master_ip, master_port, a):
    os.environ["VLLM_DP_RANK"] = str(rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(rank)
    os.environ["VLLM_DP_SIZE"] = str(dp)
    os.environ["VLLM_DP_MASTER_IP"] = master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(master_port)

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=a.model,
        tensor_parallel_size=tp,
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

    rng = np.random.default_rng(rank)
    rows = []
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
        step_ms = float(np.median(lat)) * 1000.0 / a.output_len
        rows.append({
            "batch_per_rank": B,
            "batch_global": B * dp,
            "step_ms": round(step_ms, 4),
        })
        if rank == 0:
            print(f"[d={a.delay_us:.0f}us dp={dp}] B/rank={B:>3} "
                  f"(global {B * dp:>4}): step={step_ms:8.3f} ms")

    if rank == 0:
        a.output_json.parent.mkdir(parents=True, exist_ok=True)
        a.output_json.write_text(json.dumps({
            "model": a.model,
            "data_parallel_size": dp,
            "tensor_parallel_size": tp,
            "delay_us": a.delay_us,
            "input_len": a.input_len,
            "output_len": a.output_len,
            "rows": rows,
        }, indent=2))
    time.sleep(1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    ap.add_argument("--data-parallel-size", type=int, default=4)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--batch-sizes", type=parse_int_list, default=[1, 8, 32, 64])
    ap.add_argument("--delay-us", type=float, required=True)
    ap.add_argument("--input-len", type=int, default=4)
    ap.add_argument("--output-len", type=int, default=64)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--output-json", type=Path, required=True)
    a = ap.parse_args()

    os.environ["VLLM_SELF_SPEC_EMULATE_A2A_DELAY_US"] = str(a.delay_us)
    os.environ["VLLM_SELF_SPEC_LOG_A2A_COUNTS"] = "1"

    from vllm.utils.network_utils import get_open_port

    master_ip, master_port = "127.0.0.1", get_open_port()
    dp, tp = a.data_parallel_size, a.tensor_parallel_size

    procs = []
    for rank in range(dp):
        p = Process(target=worker, args=(rank, dp, tp, master_ip, master_port, a))
        p.start()
        procs.append(p)
    code = 0
    for p in procs:
        p.join(timeout=600)
        if p.exitcode is None:
            p.kill(); code = 1
        elif p.exitcode:
            code = p.exitcode
    return code


if __name__ == "__main__":
    raise SystemExit(main())
