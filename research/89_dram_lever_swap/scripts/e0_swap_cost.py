#!/usr/bin/env python3
"""Phase 89 E0: price the DRAM-cached weight-swap mechanics (no engine).

Measures, per draft ckpt:
  (a) disk -> pinned-DRAM load (one-time cache build)
  (b) pinned-DRAM -> GPU in-place copy (the runtime swap) + GB/s
  (c) CUDA-graph replay validity after in-place copy_ -- the
      load-bearing assumption for zero-downtime swapping: a captured
      graph must read the NEW weights after param.copy_().
Then the amortization table: seconds of serving at throughput T needed
to pay for one swap at a given speedup delta.

env: E0_CKPTS (comma paths), E0_DEV (default cuda:0)
"""
import glob
import json
import os
import time
from pathlib import Path

import torch
from safetensors import safe_open

DEV = os.environ.get("E0_DEV", "cuda:0")
CKPTS = os.environ.get(
    "E0_CKPTS",
    os.path.expanduser("/data/smcho/ckpts/Qwen3-8B-W4A16-INT4") + ","
    + os.path.expanduser("/data/smcho/ckpts/Qwen3-8B-W4A8-gptq"),
).split(",")


def load_pinned(ckpt):
    """Disk -> pinned host tensors. Returns (dict, bytes, secs)."""
    t0 = time.perf_counter()
    tensors, nbytes = {}, 0
    for f in sorted(glob.glob(f"{ckpt}/*.safetensors")):
        with safe_open(f, framework="pt", device="cpu") as sf:
            for name in sf.keys():
                t = sf.get_tensor(name)
                p = torch.empty_strided(
                    t.shape, t.stride(), dtype=t.dtype, pin_memory=True)
                p.copy_(t)
                tensors[name] = p
                nbytes += t.numel() * t.element_size()
    return tensors, nbytes, time.perf_counter() - t0


def swap_to_gpu(pinned, gpu):
    """Pinned DRAM -> preallocated GPU tensors, async + one sync."""
    t0 = time.perf_counter()
    with torch.cuda.stream(torch.cuda.Stream(device=DEV)):
        for name, src in pinned.items():
            gpu[name].copy_(src, non_blocking=True)
    torch.cuda.synchronize(DEV)
    return time.perf_counter() - t0


def graph_validity_check():
    """Capture a graph over quant-shaped layers; in-place copy_ new
    weights; replay must produce the NEW result bit-exactly."""
    torch.manual_seed(0)
    # int32-packed weight + fp scale, mimicking a W4 linear's storage
    packed = torch.randint(-2**31, 2**31 - 1, (4096, 512),
                           dtype=torch.int32, device=DEV)
    scale = torch.rand(4096, 32, dtype=torch.bfloat16, device=DEV)
    x = torch.randn(64, 4096, dtype=torch.bfloat16, device=DEV)
    out = torch.empty(64, 32, dtype=torch.bfloat16, device=DEV)

    def fwd():
        # stand-in compute touching both tensors' memory
        w = packed.to(torch.bfloat16)[:, :32] * scale
        torch.matmul(x, w, out=out)

    for _ in range(3):
        fwd()
    torch.cuda.synchronize(DEV)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fwd()
    g.replay()
    torch.cuda.synchronize(DEV)
    before = out.clone()

    packed2 = torch.randint(-2**31, 2**31 - 1, (4096, 512),
                            dtype=torch.int32, device=DEV)
    scale2 = torch.rand(4096, 32, dtype=torch.bfloat16, device=DEV)
    packed.copy_(packed2)
    scale.copy_(scale2)
    g.replay()
    torch.cuda.synchronize(DEV)
    after = out.clone()

    w2 = packed2.to(torch.bfloat16)[:, :32] * scale2
    expect = torch.matmul(x, w2)
    ok_changed = not torch.equal(before, after)
    ok_exact = torch.equal(after, expect)
    return ok_changed, ok_exact


def main():
    torch.cuda.set_device(DEV)
    report = {"dev": DEV, "ckpts": {}}
    for ckpt in CKPTS:
        name = Path(ckpt).name
        pinned, nbytes, t_disk = load_pinned(ckpt)
        gb = nbytes / 1e9
        gpu = {k: torch.empty_like(v, device=DEV)
               for k, v in pinned.items()}
        times = [swap_to_gpu(pinned, gpu) for _ in range(3)]
        t_swap = sorted(times)[1]
        report["ckpts"][name] = {
            "GB": round(gb, 2),
            "disk_to_pinned_s": round(t_disk, 2),
            "swap_s": round(t_swap, 3),
            "swap_GBps": round(gb / t_swap, 1),
            "all_swap_s": [round(t, 3) for t in times],
        }
        print(f"[E0] {name}: {gb:.2f} GB, disk->pinned {t_disk:.1f}s "
              f"(one-time), DRAM->GPU swap {t_swap*1000:.0f} ms "
              f"({gb/t_swap:.0f} GB/s)", flush=True)
        del gpu
        torch.cuda.empty_cache()

    ok_changed, ok_exact = graph_validity_check()
    report["graph_replay_sees_inplace_copy"] = ok_changed
    report["graph_replay_bitexact_new_weights"] = ok_exact
    print(f"[E0] graph validity: replay-sees-new-weights={ok_changed} "
          f"bit-exact={ok_exact}", flush=True)

    # amortization: swap pays when dS * T * t_serve > T * t_swap
    # -> t_serve > t_swap / dS
    print("[E0] amortization (seconds of serving to pay one swap):",
          flush=True)
    for nm, c in report["ckpts"].items():
        for dS in (0.03, 0.10, 0.30):
            print(f"       {nm} at +{int(dS*100)}%: "
                  f"{c['swap_s']/dS:.1f}s", flush=True)
    out = Path(__file__).resolve().parents[1] / "data" / "e0_swap_cost.json"
    out.write_text(json.dumps(report, indent=1))
    print("[E0] saved ->", out, flush=True)


if __name__ == "__main__":
    main()
