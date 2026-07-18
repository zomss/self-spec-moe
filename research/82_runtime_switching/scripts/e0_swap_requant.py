#!/usr/bin/env python3
"""E0 microbenches: (a) draft-swap upload cost, (b) C9 CPU KV requant.

(a) Draft weight swap: time safetensors -> CPU -> GPU for the W4 8B
    draft ckpt (warm page cache), plus a pinned-blob H2D roofline.
    Memory-rent alternative: both-resident sizes printed.
(b) C9 (DIRECTION 3.2): CPU-side KV requant bf16 -> e4m3 (pool
    semantics: clamp +-448, scale 1.0) + H2D upload of the fp8 pool,
    timed per layer and extrapolated to the b8/16k Qwen3-8B cell
    (36 layers x 2 x 131072 tok x 8 heads x 128 dim).
"""
import glob
import os
import time

import torch

CKPT = os.path.expanduser("~/ckpts/Qwen3-8B-W4A16-INT4")


def bench_swap():
    from safetensors import safe_open
    files = sorted(glob.glob(CKPT + "/*.safetensors"))
    t0 = time.perf_counter()
    cpu = []
    for f in files:
        with safe_open(f, framework="pt") as sf:
            for k in sf.keys():
                cpu.append(sf.get_tensor(k))
    t_load = time.perf_counter() - t0
    nbytes = sum(t.numel() * t.element_size() for t in cpu)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    gpu = [t.cuda(non_blocking=False) for t in cpu]
    torch.cuda.synchronize()
    t_h2d = time.perf_counter() - t0
    del gpu, cpu
    torch.cuda.empty_cache()
    blob = torch.empty(2 << 30, dtype=torch.uint8, pin_memory=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = blob.cuda(non_blocking=False)
    torch.cuda.synchronize()
    t_pin = time.perf_counter() - t0
    gb = nbytes / 2**30
    print(f"[swap] ckpt {gb:.2f} GiB: file->cpu {t_load:.2f}s, "
          f"cpu->gpu {t_h2d:.2f}s ({gb/t_h2d:.1f} GiB/s); "
          f"pinned roofline {2/t_pin:.1f} GiB/s "
          f"-> pinned-staged swap est {gb/(2/t_pin):.2f}s", flush=True)
    return dict(ckpt_gib=round(gb, 2), file_to_cpu_s=round(t_load, 2),
                h2d_s=round(t_h2d, 2),
                pinned_gibps=round(2 / t_pin, 1))


def bench_requant(layers=36, tok=131072, kvh=8, hd=128, sample=4):
    per_layer = []
    for _ in range(sample):
        kv = torch.randn(2, tok, kvh, hd, dtype=torch.bfloat16)
        t0 = time.perf_counter()
        q = kv.clamp(-448, 448).to(torch.float8_e4m3fn)
        t_q = time.perf_counter() - t0
        t0 = time.perf_counter()
        _ = q.cuda(non_blocking=False)
        torch.cuda.synchronize()
        t_u = time.perf_counter() - t0
        per_layer.append((t_q, t_u))
        del kv, q
    tq = sum(a for a, _ in per_layer) / sample
    tu = sum(b for _, b in per_layer) / sample
    gib_l = 2 * tok * kvh * hd * 2 / 2**30
    total = (tq + tu) * layers
    print(f"[requant] per layer ({gib_l:.2f} GiB bf16): cpu-quant {tq:.2f}s"
          f" + upload {tu:.2f}s -> b8/16k cell ({layers}L, "
          f"{gib_l*layers:.0f} GiB) = {total:.1f}s total", flush=True)
    return dict(per_layer_quant_s=round(tq, 3), per_layer_h2d_s=round(tu, 3),
                cell_total_s=round(total, 1))


if __name__ == "__main__":
    import json
    out = {"swap": bench_swap(), "requant_c9": bench_requant()}
    p = os.path.join(os.path.dirname(__file__), "../data/e0_swap_requant.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print("saved ->", p)
