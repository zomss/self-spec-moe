#!/usr/bin/env python3
"""All-to-all latency microbenchmark across local GPUs.

Run once per transport (NVLink default; NCCL_P2P_DISABLE=1 etc for a slow proxy):
the draft step pays the intra-node (NVLink) all-to-all; the inter-node all-to-all
(approximated by the slow transport / modeled separately) is what the draft saves.
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# per-peer message sizes in bytes (decode MoE dispatch is small, latency-bound)
SIZES = [512, 2048, 8192, 32768, 131072, 524288]


def worker(rank, world, iters, warmup, label, out_path, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")
    out = []
    for nbytes in SIZES:
        elems_per_peer = max(1, nbytes // 2)  # bfloat16
        send = torch.randn(elems_per_peer * world, dtype=torch.bfloat16, device=dev)
        recv = torch.empty_like(send)
        for _ in range(warmup):
            dist.all_to_all_single(recv, send)
        torch.cuda.synchronize()
        dist.barrier()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            dist.all_to_all_single(recv, send)
        e.record()
        torch.cuda.synchronize()
        us = s.elapsed_time(e) / iters * 1000.0
        out.append({"per_peer_bytes": nbytes, "latency_us": round(us, 3)})
        if rank == 0:
            print(f"[{label}] per-peer {nbytes:>7}B  ->  {us:8.2f} us")
    if rank == 0:
        with open(out_path, "w") as f:
            json.dump({"label": label, "world": world, "results": out}, f, indent=2)
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, default=8)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--port", type=int, default=29577)
    a = ap.parse_args()
    print(f"=== all-to-all bench: label={a.label} world={a.world} "
          f"P2P_DISABLE={os.environ.get('NCCL_P2P_DISABLE')} "
          f"SHM_DISABLE={os.environ.get('NCCL_SHM_DISABLE')} ===")
    mp.spawn(worker, args=(a.world, a.iters, a.warmup, a.label, a.out, a.port), nprocs=a.world)


if __name__ == "__main__":
    main()
