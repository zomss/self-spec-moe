#!/usr/bin/env python3
"""Stage B2b-pragmatic: in-loop overhead -> overhead-accounted integrated tokens/s.

The integrated lockstep tokens/s = composed from measured parts (B2a comm-free draft
step S_draft + Stage A comm-bound verify step S_verify + B1 acceptance), plus the
per-cycle in-loop overhead the composition ignores: rank-local rejection sampling and
the verify-warmed cache update. This measures that overhead (GPU time at realistic
sizes) and folds it in, avoiding the week-scale distributed step-level driver.

Overhead is rank-local, so a single-GPU measurement of one rank's per-cycle cost is
representative. We upper-bound it (full softmax over vocab + naive per-layer cache
update loop).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

# measured engine step times (ms) from B2a / Stage A, keyed by PER-RANK batch
# (socket fabric, dp=8). S_draft = skip-A2A; S_verify = full-EP socket.
STEP_MS = {
    1: {"S_draft": 18.62, "S_verify": 92.62},
    4: {"S_draft": 19.75, "S_verify": 117.56},
    16: {"S_draft": 19.20, "S_verify": 138.52},
}
BETAS = {"0.82": 0.82, "0.92": 0.92}


def cuda_time(fn, iters, warmup, device):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize(device)
    return start.elapsed_time(end) / iters  # ms/call


def reject_fn(k, B, V, device):
    qlog = torch.randn(k, B, V, device=device)
    plog = torch.randn(k, B, V, device=device)

    def run():
        q = torch.softmax(qlog, -1)
        p = torch.softmax(plog, -1)
        t = q.argmax(-1)
        pr = p.gather(-1, t.unsqueeze(-1)).squeeze(-1)
        qr = q.gather(-1, t.unsqueeze(-1)).squeeze(-1)
        ratio = (pr / qr.clamp_min(1e-9)).clamp_max(1.0)
        _ = torch.rand(k, B, device=device) < ratio
        resid = (p - q).clamp_min(0)
        resid = resid / resid.sum(-1, keepdim=True).clamp_min(1e-9)
        _ = torch.multinomial(resid.view(-1, V), 1)
    return run


def cache_fn(L, B, top_k, E, C, device):
    ids = torch.randint(0, E, (L, B, top_k), device=device)

    def run():
        for layer in range(L):
            freq = torch.bincount(ids[layer].view(-1), minlength=E)
            _ = torch.topk(freq, C)
    return run


def tokens_per_cycle(k, b):
    return 1.0 + b * (1 - b ** k) / (1 - b)


def best(k_list, sd, sv, beta, extra=0.0):
    bsp, bk = 0.0, 0
    for k in k_list:
        sp = tokens_per_cycle(k, beta) * sv / (k * sd + sv + extra)
        if sp > bsp:
            bsp, bk = sp, k
    return bsp, bk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--vocab", type=int, default=151936)
    ap.add_argument("--layers", type=int, default=48)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--experts", type=int, default=128)
    ap.add_argument("--cache-c", type=int, default=64)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--batches", default="1,4,16")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--output-json", type=Path, required=True)
    a = ap.parse_args()
    dev = a.device
    Bs = [int(x) for x in a.batches.split(",")]
    KS = [1, 2, 4, 6, 8]

    rows = []
    for B in Bs:
        t_rej = cuda_time(reject_fn(a.k, B, a.vocab, dev), a.iters, a.warmup, dev)
        t_cache = cuda_time(
            cache_fn(a.layers, B, a.top_k, a.experts, a.cache_c, dev),
            a.iters, a.warmup, dev,
        )
        overhead = t_rej + t_cache
        sd = STEP_MS[B]["S_draft"]
        sv = STEP_MS[B]["S_verify"]
        row = {"batch_per_rank": B, "t_reject_ms": round(t_rej, 4),
               "t_cache_ms": round(t_cache, 4), "overhead_ms": round(overhead, 4),
               "S_draft": sd, "S_verify": sv}
        for label, beta in BETAS.items():
            sp0, k0 = best(KS, sd, sv, beta, 0.0)
            sp1, k1 = best(KS, sd, sv, beta, overhead)
            cyc = k1 * sd + sv
            row[f"speedup_b{label}_no_oh"] = round(sp0, 3)
            row[f"speedup_b{label}_with_oh"] = round(sp1, 3)
            row[f"overhead_frac_b{label}"] = round(overhead / (cyc + overhead), 5)
        rows.append(row)
        print(f"B/rank={B}: t_reject={t_rej:.3f}ms t_cache={t_cache:.3f}ms "
              f"overhead={overhead:.3f}ms (cycle~{a.k*sd+sv:.0f}ms) | "
              f"b0.92 speedup {row['speedup_b0.92_no_oh']}->{row['speedup_b0.92_with_oh']}"
              f" (oh {row['overhead_frac_b0.92']*100:.3f}%)")

    out = {"vocab": a.vocab, "layers": a.layers, "k": a.k, "rows": rows}
    a.output_json.parent.mkdir(parents=True, exist_ok=True)
    a.output_json.write_text(json.dumps(out, indent=2))
    print(f"[saved] {a.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
