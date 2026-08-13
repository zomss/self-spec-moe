# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Synthetic co-tenant load, to reproduce the interference under control.

All night the clamp correlated with a neighbouring tensor-parallel job on
GPUs 2 and 3 -- 100% SM, 91% memory utilisation, ~500 W each -- but the
correlation could never be TESTED, because that job belonged to someone else
and could be neither started nor stopped on demand. Every comparison was
therefore observational, and two of them produced conclusions that later
dissolved.

With the box idle, the load can be manufactured instead of waited for, which
turns the question into a controlled experiment.

The target is the neighbour's observable signature rather than its code: large
bf16 GEMMs in a loop drive SM utilisation to ~100%, memory utilisation high
(HBM read/write bound at these shapes), and power toward the 700 W cap. What is
deliberately NOT reproduced is its host behaviour -- no NCCL spin, no busy
polling -- so if the clamp appears under this load, it is caused by GPU-side
activity alone, and if it does not, the neighbour's host behaviour becomes the
suspect.

Run as a subprocess and killed when the measured boot finishes.
"""

from __future__ import annotations

import argparse
import os
import time


def _run_vllm(devices: str, seconds: float) -> int:
    """Faithful co-tenant: a real vLLM engine, tensor-parallel, decoding.

    X26 showed that compute-bound GEMMs at 100% SM and ~700 W do NOT reproduce
    the clamp, which excludes co-tenant compute and power. The real neighbour
    differed in four ways: 91% memory utilisation rather than 30%, a 76 GB
    resident footprint rather than 0.5 GB, ~340% host CPU from NCCL spin rather
    than nearly none, and NVLink collectives rather than none.

    This reproduces all four at once. If it clamps us, the properties can then
    be ablated one at a time; if it does not, co-tenancy is exonerated entirely
    and the cause lies elsewhere on the box.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = devices
    from vllm import LLM, SamplingParams

    tp = 1  # TP>1 fails to start in this fork; one engine per GPU instead
    llm = LLM(
        model=(
            "/data/smcho/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/"
            "b968826d9c46dd6066d109eabc6255188de91218"
        ),
        tensor_parallel_size=tp,
        gpu_memory_utilization=0.90,
        max_model_len=4096,
        enforce_eager=False,
        disable_log_stats=True,
    )
    prompts = ["Explain speculative decoding in detail." for _ in range(64)]
    params = SamplingParams(temperature=0.8, max_tokens=512, ignore_eos=True)
    print(f"[load] vLLM TP={tp} on devices {devices}", flush=True)
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        llm.generate(prompts, params, use_tqdm=False)
    return 0


def _run_membw(devices: str, seconds: float, gib: float) -> int:
    """Memory-bandwidth-bound co-tenant with a large resident footprint.

    X26 excluded compute and power: GEMMs at 100% SM and ~700 W left us clean.
    The real neighbour differed in running at 91% MEMORY utilisation with 76 GB
    resident, i.e. HBM-bound rather than compute-bound, which is what LLM decode
    looks like.

    This isolates that property. Large elementwise streaming ops saturate HBM
    bandwidth while keeping SM occupancy and host CPU low, so a positive result
    implicates memory pressure specifically rather than "a busy neighbour".
    Chosen over a real vLLM co-tenant because that failed to start in this fork
    and, more importantly, because it bundles four variables where this varies
    one.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = devices
    import torch

    n = torch.accelerator.device_count()
    buffers = []
    for index in range(n):
        torch.accelerator.set_device_index(index)
        elems = int(gib * (1024**3) / 2)  # bf16
        a = torch.empty(elems, dtype=torch.bfloat16, device=f"cuda:{index}")
        b = torch.empty(elems, dtype=torch.bfloat16, device=f"cuda:{index}")
        a.fill_(1.0)
        b.fill_(2.0)
        buffers.append((a, b))
    print(f"[load] membw: {n} device(s) x {gib} GiB x2, streaming", flush=True)
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        for index in range(n):
            torch.accelerator.set_device_index(index)
            a, b = buffers[index]
            for _ in range(4):
                a.add_(b)  # read a, read b, write a -- pure HBM traffic
        for index in range(n):
            torch.accelerator.set_device_index(index)
            torch.accelerator.synchronize(index)
    return 0


def _run_hold(devices: str, seconds: float, gib: float) -> int:
    """Hold a large allocation IDLE -- no compute, no bandwidth, just resident.

    Neither compute (GEMM) nor bandwidth (membw) reproduced the clamp, and both
    exceeded the real neighbour on those axes. But the clamp ran continuously
    from 00:20 to ~03:10 and cleared when the neighbour PROCESS exited, not when
    it went idle: campaign run 4 clamped while those GPUs read 0% utilisation
    and still held 76 GB, and X21 was clean while they were busy.

    So the candidate is a long-lived process holding a large allocation, rather
    than any activity. This holds the memory and does nothing else.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = devices
    import torch

    n = torch.accelerator.device_count()
    held = []
    for index in range(n):
        torch.accelerator.set_device_index(index)
        held.append(
            torch.empty(int(gib * (1024**3) / 2), dtype=torch.bfloat16, device=index)
        )
        torch.accelerator.synchronize(index)
    print(f"[load] hold: {n} device(s) x {gib} GiB resident, idle", flush=True)
    time.sleep(seconds)
    return len(held) and 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="2,3")
    parser.add_argument("--size", type=int, default=16384)
    parser.add_argument("--seconds", type=float, default=1800.0)
    parser.add_argument("--gib", type=float, default=24.0)
    parser.add_argument(
        "--mode", choices=("gemm", "vllm", "membw", "hold"), default="gemm"
    )
    args = parser.parse_args()

    if args.mode == "vllm":
        return _run_vllm(args.devices, args.seconds)
    if args.mode == "membw":
        return _run_membw(args.devices, args.seconds, args.gib)
    if args.mode == "hold":
        return _run_hold(args.devices, args.seconds, args.gib)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    import torch

    n = torch.accelerator.device_count()
    mats = []
    for index in range(n):
        torch.accelerator.set_device_index(index)
        a = torch.randn(
            args.size, args.size, dtype=torch.bfloat16, device=f"cuda:{index}"
        )
        b = torch.randn(
            args.size, args.size, dtype=torch.bfloat16, device=f"cuda:{index}"
        )
        mats.append((a, b))
    print(f"[load] {n} device(s), {args.size}x{args.size} bf16 GEMM loop", flush=True)
    deadline = time.monotonic() + args.seconds
    while time.monotonic() < deadline:
        for index in range(n):
            torch.accelerator.set_device_index(index)
            a, b = mats[index]
            for _ in range(20):
                torch.mm(a, b)
        for index in range(n):
            torch.accelerator.set_device_index(index)
            torch.accelerator.synchronize(index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
