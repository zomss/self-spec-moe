"""2-node NCCL collective latency bench at decode-sized payloads.

Launch with torchrun on both nodes (see run_nccl_bench.sh). Measures
all_to_all_single and all_reduce per-collective GPU time (same-stream
serialization => event time over N back-to-back ops = per-op latency,
including network wait).
"""

import os

import torch
import torch.distributed as dist


def bench(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters  # us per op


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    results = []

    # all_to_all_single: total send buffer per rank (bytes), bf16
    for size_bytes in [16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864]:
        elems = size_bytes // 2
        inp = torch.randn(elems, dtype=torch.bfloat16, device="cuda")
        out = torch.empty_like(inp)
        iters = 100 if size_bytes <= 4194304 else 30
        us = bench(lambda: dist.all_to_all_single(out, inp), 20, iters)
        t = torch.tensor([us], device="cuda")
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        results.append(("all_to_all_single", size_bytes, t.item()))

    # small all_reduce (DP sync barrier proxy)
    for size_bytes in [8, 4096, 1048576]:
        elems = max(size_bytes // 2, 1)
        buf = torch.randn(elems, dtype=torch.bfloat16, device="cuda")
        us = bench(lambda: dist.all_reduce(buf), 20, 100)
        t = torch.tensor([us], device="cuda")
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        results.append(("all_reduce", size_bytes, t.item()))

    if rank == 0:
        print(f"# world={world} ranks, per-collective latency (max across ranks)")
        print("op,total_bytes_per_rank,us_per_op")
        for op, size, us in results:
            print(f"{op},{size},{us:.1f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
