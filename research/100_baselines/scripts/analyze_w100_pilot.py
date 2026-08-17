#!/usr/bin/env python3
"""Pilot analysis: drain-aware KV feasibility + length summary.

The pilot runner's inline feasibility used a static bound
(p95 total tokens x batch), which assumes every request holds its
p95-length KV simultaneously. That contradicts the protocol it serves:
with EOS respected, requests finish at different times and FREE their KV,
so the binding quantity is the PEAK of concurrent KV over the drain.

Model: b requests admitted together, prompts resident from t=0, one
decode token per alive request per step, request i frees its blocks
after o_i steps:

    KV(t) = sum_{i: o_i > t} (p_i + t) * BYTES_PER_TOKEN

Peak over t, with measured (p_i, o_i) tiled cyclically when b exceeds
the pilot sample count. "ok" means the whole batch can be genuinely
concurrent inside the KV pool -- vLLM would otherwise queue or preempt,
making the nominal batch a fiction.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PHASE = Path(__file__).resolve().parent.parent
PILOT = PHASE / "data" / "pilot" / "pilot_lengths.json"

KV_BYTES_PER_TOKEN = 36 * 8 * 128 * 2 * 2
POOL_BYTES = 80e9 * 0.90 - 16.4e9      # HBM x util - BF16 weights
BATCHES = (1, 8, 16, 32, 64)


def peak_kv_bytes(pairs: list[tuple[int, int]]) -> tuple[int, int]:
    """(peak_bytes, argmax_t) for one synchronized-admission drain."""
    horizon = max(o for _, o in pairs)
    # Piecewise-linear between completion times: evaluate at each
    # completion boundary and at t=0.
    times = sorted({0, horizon} | {o for _, o in pairs})
    best, best_t = 0, 0
    for t in times:
        for tt in (max(0, t - 1), t):
            kv = sum(p + tt for p, o in pairs if o > tt)
            if kv > best:
                best, best_t = kv, tt
    return best * KV_BYTES_PER_TOKEN, best_t


def main() -> None:
    rec = json.loads(PILOT.read_text())
    out = {"pool_bytes": POOL_BYTES,
           "kv_bytes_per_token": KV_BYTES_PER_TOKEN, "cells": {}}
    for cell, e in rec["cells"].items():
        pairs = [(r["prompt_toks"], r["out_toks"]) for r in e["requests"]]
        outs = sorted(o for _, o in pairs)
        q = lambda p: outs[min(len(outs) - 1, int(p * len(outs)))]
        caps = sum(r["finish"] == "length" for r in e["requests"])
        cellrep = {
            "n": len(pairs),
            "out_median": q(0.5), "out_p95": q(0.95), "out_max": outs[-1],
            "cap_hits": caps, "cap": e["max_tokens"],
            "batches": {},
        }
        for b in BATCHES:
            tiled = [pairs[i % len(pairs)] for i in range(b)]
            peak, at = peak_kv_bytes(tiled)
            cellrep["batches"][b] = {
                "peak_kv_gb": round(peak / 2**30, 1),
                "peak_at_step": at,
                "verdict": "ok" if peak <= POOL_BYTES else "infeasible",
            }
        out["cells"][cell] = cellrep
        verd = {b: v["verdict"] for b, v in cellrep["batches"].items()}
        peaks = {b: v["peak_kv_gb"] for b, v in cellrep["batches"].items()}
        print(f"[{cell}] out med/p95/max = {q(0.5)}/{q(0.95)}/{outs[-1]}"
              f"  cap_hits {caps}/{len(pairs)}")
        print(f"    peak KV GB {peaks}")
        print(f"    verdicts   {verd}")
    dest = PHASE / "data" / "pilot" / "pilot_feasibility.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"-> {dest}")


if __name__ == "__main__":
    main()
