"""OV0a: can side-stream compute proceed during PCIe-SHM NCCL collectives?

torchrun x8 physics probe (no vLLM). Per "step":
  comm  = OV_COLLECTIVES/2 x (all_gather_into_tensor + reduce_scatter_tensor) of
          [tokens/rank, hidden] bf16 on the MAIN stream (the verify's A2A shape:
          48 layers x dispatch+combine = 96 collectives/step), or, in the
          emulated-latency variant, OV_COLLECTIVES x torch.cuda._sleep(delay).
  comp  = N x [4096x4096]@[4096x4096] bf16 GEMMs calibrated to OV_TARGET_MS
          (~one K=2 draft chain, ~32 ms) on a SIDE stream.

Measures T_comm, T_comp, T_both (comm on main + comp on side, join) and reports
hidden = (T_comm + T_comp - T_both) / min(T_comm, T_comp)  in [0..1].

Env: OV_TOKENS_PER_RANK (24), OV_HIDDEN (2048), OV_COLLECTIVES (96),
OV_TARGET_MS (32), OV_SLEEP_US (0 -> skip sleep variant), OV_ITERS (20),
OV_OUT (json path, rank0).
"""

import json
import os
import time

import torch
import torch.distributed as dist


def env_int(k, d):
    return int(os.environ.get(k, str(d)))


def env_float(k, d):
    return float(os.environ.get(k, str(d)))


def calibrate_cycles_per_us() -> float:
    torch.cuda._sleep(1_000_000)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    n = 200_000_000
    s.record()
    torch.cuda._sleep(n)
    e.record()
    torch.cuda.synchronize()
    ms = s.elapsed_time(e)
    return n / (ms * 1000.0) if ms > 0 else 0.0


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)

    tokens = env_int("OV_TOKENS_PER_RANK", 24)
    hidden = env_int("OV_HIDDEN", 2048)
    n_coll = env_int("OV_COLLECTIVES", 96)
    target_ms = env_float("OV_TARGET_MS", 32.0)
    sleep_us = env_float("OV_SLEEP_US", 0.0)
    iters = env_int("OV_ITERS", 20)
    warmup = 5

    local = torch.randn(tokens, hidden, dtype=torch.bfloat16, device=dev)
    gathered = torch.empty(world * tokens, hidden, dtype=torch.bfloat16, device=dev)
    rs_in = torch.randn(world * tokens, hidden, dtype=torch.bfloat16, device=dev)
    rs_out = torch.empty(tokens, hidden, dtype=torch.bfloat16, device=dev)

    def comm_step():
        for _ in range(n_coll // 2):
            dist.all_gather_into_tensor(gathered, local)
            dist.reduce_scatter_tensor(rs_out, rs_in)

    cycles_per_us = calibrate_cycles_per_us()

    def sleep_step():
        for _ in range(n_coll):
            torch.cuda._sleep(int(sleep_us * cycles_per_us))

    # Calibrate the GEMM load to ~target_ms (one draft chain).
    a = torch.randn(4096, 4096, dtype=torch.bfloat16, device=dev)
    b = torch.randn(4096, 4096, dtype=torch.bfloat16, device=dev)
    for _ in range(10):
        torch.mm(a, b)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        torch.mm(a, b)
    torch.cuda.synchronize()
    per_gemm_ms = (time.perf_counter() - t0) * 1000 / 50
    n_gemm = max(1, int(round(target_ms / per_gemm_ms)))

    def comp_step():
        for _ in range(n_gemm):
            torch.mm(a, b)

    side = torch.cuda.Stream()

    def timed(fn_main, fn_side=None):
        ts = []
        for _ in range(warmup + iters):
            dist.barrier()
            torch.cuda.synchronize()
            t = time.perf_counter()
            if fn_side is not None:
                with torch.cuda.stream(side):
                    fn_side()
            fn_main()
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t) * 1000)
        ts = sorted(ts[warmup:])
        return ts[len(ts) // 2]

    res = {
        "world": world,
        "tokens_per_rank": tokens,
        "hidden": hidden,
        "n_coll": n_coll,
        "n_gemm": n_gemm,
        "per_gemm_ms": per_gemm_ms,
    }

    res["t_comp"] = timed(comp_step)
    res["t_comm"] = timed(comm_step)
    res["t_both_comm"] = timed(comm_step, comp_step)
    res["hidden_frac_comm"] = (
        res["t_comm"] + res["t_comp"] - res["t_both_comm"]
    ) / min(res["t_comm"], res["t_comp"])

    if sleep_us > 0:
        res["sleep_us"] = sleep_us
        res["t_sleep"] = timed(sleep_step)
        res["t_both_sleep"] = timed(sleep_step, comp_step)
        res["hidden_frac_sleep"] = (
            res["t_sleep"] + res["t_comp"] - res["t_both_sleep"]
        ) / min(res["t_sleep"], res["t_comp"])

    if rank == 0:
        print(json.dumps(res, indent=2))
        out = os.environ.get("OV_OUT")
        if out:
            with open(out, "w") as f:
                json.dump(res, f, indent=2)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
