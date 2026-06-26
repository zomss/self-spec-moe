#!/usr/bin/env python3
"""Comm-bound decode step time: attention-DP + EP, NVLink vs forced-PCIe.

Stage A regime confirmation (Phase 24). Same attention-DP + EP decode as Phase 20's
bench_dp_injection.py (the hooked AgRs all_gatherv/reduce_scatterv path), but instead
of injecting a delay we toggle the *real* fabric: `NCCL_P2P_DISABLE=1` forces the EP
all-to-all off NVLink onto host-staged PCIe, emulating a comm-bound (no-NVLink) box.

Compute is identical with P2P on/off (same kernels, same data), so:
    S_on  (NVLink)        ~= compute-only
    S_off (P2P disabled)  =  compute + PCIe comm
    f_off = (S_off - S_on) / S_off          (the exposed comm fraction)
and a local-routing draft that skips the all-to-all has step ~= S_on even on the
comm-bound box. Run this for P2P in {0,1}; compose_speedup.py does the rest.

Set NCCL_P2P_DISABLE in the environment before launching; children inherit it.
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
            print(f"[p2p_disable={os.environ.get('NCCL_P2P_DISABLE','0')} dp={dp}] "
                  f"B/rank={B:>3} (global {B * dp:>4}): step={step_ms:8.3f} ms")

    if rank == 0:
        a.output_json.parent.mkdir(parents=True, exist_ok=True)
        a.output_json.write_text(json.dumps({
            "model": a.model,
            "tag": a.tag,
            "nccl_p2p_disable": os.environ.get("NCCL_P2P_DISABLE", "0"),
            "nccl_shm_disable": os.environ.get("NCCL_SHM_DISABLE", "0"),
            "data_parallel_size": dp,
            "tensor_parallel_size": tp,
            "input_len": a.input_len,
            "output_len": a.output_len,
            "rows": rows,
        }, indent=2))
    time.sleep(1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--data-parallel-size", type=int, default=8)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--batch-sizes", type=parse_int_list, default=[1, 4, 16, 64])
    ap.add_argument("--input-len", type=int, default=4)
    ap.add_argument("--output-len", type=int, default=64)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--output-json", type=Path, required=True)
    a = ap.parse_args()

    # research hooks off; we measure the real fabric, not an injected delay
    os.environ["VLLM_SELF_SPEC_EMULATE_A2A_DELAY_US"] = "0"
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
        p.join(timeout=900)
        if p.exitcode is None:
            p.kill(); code = 1
        elif p.exitcode:
            code = p.exitcode
    return code


if __name__ == "__main__":
    raise SystemExit(main())
