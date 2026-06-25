#!/usr/bin/env python3
"""Derive N, exposed-A2A fraction f, and the beta-weighted speedup envelope from
the dp4 injection sweep.

Lockstep cycle = k local-draft steps (collective-free: pays NO injected delay) +
1 full-EP verify step (pays it). With one-step acceptance beta:

  E[accepted](k) = (1 - beta^(k+1)) / (1 - beta)
  speedup(k,f)   = E[accepted] / ((k+1) - k*f)        f = N*d / (S_base + N*d)

f->1 (comm-bound) => speedup -> E[accepted]; f->0 => <=1 (no benefit). The draft
removes the exposed inter-node A2A; verify keeps it lossless.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

D = Path(__file__).parent / "data"
DELAYS = [0, 163, 320]
# Data file prefix (dp4 = Qwen3; gptoss_dp4 = GPT-OSS). Override via argv[1].
PREFIX = sys.argv[1] if len(sys.argv) > 1 else "dp4"
# Measured one-step acceptance from prior phases (the draft-quality axis).
BETAS = {
    "affinity G=2 (0.85)": 0.85,
    "EPLB G=4 R~2x (0.78)": 0.78,
    "FP8 draft (0.95)": 0.95,
}


def e_accept(beta, k):
    return (1 - beta ** (k + 1)) / (1 - beta)


def best_k(beta, f, kmax=12):
    best = (0, 1.0)  # k=0 -> no spec -> speedup 1
    for k in range(1, kmax + 1):
        s = e_accept(beta, k) / ((k + 1) - k * f)
        if s > best[1]:
            best = (k, s)
    return best


def main():
    data = {d: {r["batch_per_rank"]: r for r in
                json.loads((D / f"{PREFIX}_d{d}.json").read_text())["rows"]}
            for d in DELAYS}
    meta = json.loads((D / f"{PREFIX}_d0.json").read_text())
    dp = meta["data_parallel_size"]
    batches = sorted(data[0])

    print(f"=== Measured step time S(B,d), dp={dp} ({meta['model']}, EP via AgRs) ===")
    print(f"{'B/rank':>6} {'global':>6} | {'S(0)':>7} {'S(163)':>7} {'S(320)':>7} | "
          f"{'N':>5} | {'f@163':>6} {'f@320':>6}")
    print("-" * 70)
    rows = []
    for b in batches:
        s0 = data[0][b]["step_ms"]
        s163 = data[163][b]["step_ms"]
        s320 = data[320][b]["step_ms"]
        n = (s320 - s0) / (320 / 1000.0)  # collectives/step from the larger delay
        f163 = (s163 - s0) / s163
        f320 = (s320 - s0) / s320
        g = data[0][b]["batch_global"]
        rows.append((b, g, s0, f163, f320, n))
        print(f"{b:>6} {g:>6} | {s0:7.2f} {s163:7.2f} {s320:7.2f} | {n:5.0f} | "
              f"{f163:6.3f} {f320:6.3f}")

    # Break-even: speedup>1 at k=1 iff f > 1-beta, i.e. exposed per-collective
    # latency d* = (1-beta)*S_base / (N*beta). With ~145 collectives/step the bar
    # is tiny. (Uses the low-batch point, global=4.)
    b0, g0, s0, _, _, n0 = rows[0]
    print(f"\n=== Break-even exposed-A2A per collective (low batch, global {g0}, "
          f"S_base={s0:.2f}ms, N={n0:.0f}) ===")
    print(f"{'beta':>22} | {'break-even f*':>13} | {'d* per collective':>18}")
    print("-" * 60)
    for blabel, beta in BETAS.items():
        fstar = 1 - beta
        dstar_us = fstar * s0 / (n0 * beta) * 1000.0  # ms->us
        print(f"{blabel:>22} | {fstar:13.3f} | {dstar_us:14.1f} us")

    print(f"\n=== Speedup vs exposed per-collective latency d (low batch, global {g0}) ===")
    hdr = " | ".join(f"{lbl.split(' (')[0]:>12}" for lbl in BETAS)
    print(f"{'d (us)':>7} | {'f':>6} | " + hdr)
    print("-" * 64)
    for d_us in (10, 25, 50, 100, 163, 250, 320):
        nd = n0 * (d_us / 1000.0)
        f = nd / (s0 + nd)
        cells = []
        for beta in BETAS.values():
            _, s = best_k(beta, f)
            cells.append(f"{s:12.2f}")
        print(f"{d_us:>7} | {f:6.3f} | " + " | ".join(cells))

    print("\n=== beta-weighted speedup envelope (best k) ===")
    print("Per (batch, d, beta): exposed f, optimal cycle length k*, lossless speedup")
    for b, g, s0, f163, f320, n in rows:
        for dlabel, f in (("163us", f163), ("320us", f320)):
            print(f"\n  B/rank={b} (global {g}), d={dlabel}, f={f:.3f}:")
            for blabel, beta in BETAS.items():
                k, s = best_k(beta, f)
                print(f"    {blabel:<24} -> k*={k:>2}  speedup={s:4.2f}x")


if __name__ == "__main__":
    raise SystemExit(main())
