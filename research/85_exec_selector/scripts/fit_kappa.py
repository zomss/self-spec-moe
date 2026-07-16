#!/usr/bin/env python3
"""kappa(M): the fp8-dequant small-M penalty as a function, not a constant.

Two recorded mispricings trace to constant-kappa: the MLA challenger's
native-fp8 R and the flip realization's fp8-vs-bf16 inversion between b4
and b32. Model: kappa(M) = k0 * max(0, 1 - M/M0) (linear ramp, zero above
M0), M = mean tokens per expert per forward.

Evidence rows (source noted; clean cells only, S4-flagged excluded):
- MLA native-fp8 serve R at 2k (M = b*6/64): the strongest small-M signal
  (R 1.90 at M=0.375 decaying to ~1.04 at M=3).
- The flip realization pair (83-E2c): fp8-full vs bf16-partial at b4/b32
  under DP4 (M = (b/4)*8/128): fp8 loses 11% at M=0.0625, WINS at M=0.5
  -- the crossover pins the ramp against the byte advantage.
"""

import json
from pathlib import Path

PHASE = Path(__file__).resolve().parents[1]

# (M, kappa_implied_Tt_units, weight, source)
# kappa_implied = R_meas - R_bytes, with R_bytes from the additive model's
# byte terms (fp8 native ~0.93 at 2k where weights dominate the draft read;
# MLA target Tt basis). For the flip pair, the DELTA between realizations
# isolates kappa: (R_fp8 - R_bf16partial) + byte_gap (fp8 reads ~0.5x the
# bytes of bf16-partial at saturation -> byte_gap ~ -0.10 at b32).
ROWS = [
    # MLA native fp8, 2k serve row (fig1: 1.90 / 0.94? no: b4 1.90, b8 0.94, b32 1.05)
    (0.375, 1.90 - 0.93, 0.0, "mla ds_fp8block b4/2k [S4-class outlier: b8 neighbor implies ~0]"),
    (0.750, 0.94 - 0.93, 0.0, "mla ds_fp8block b8/2k [context]"),
    (3.000, 1.05 - 0.93, 0.0, "mla ds_fp8block b32/2k [context]"),
    # MoE native fp8 2k (M = b*8/128 under DP4 serve: b4->0.0625? serve is
    # DP4 so per-rank b/4; use M = (b/4)*8/128)
    (0.0625, 1.036 - 0.93, 0.2, "moe m_fp8block b4/2k (DP4) [noisy: +-10%]"),
    (0.125, 1.010 - 0.93, 0.2, "moe m_fp8block b8/2k [noisy]"),
    (0.500, 1.029 - 0.93, 0.2, "moe m_fp8block b32/2k [noisy]"),
    # flip realization delta (chain, DP4): fp8-full minus bf16-partial
    # speedup ratios converted to R-delta at gamma=2:
    # b4: fp8 0.93x vs bf16 1.03x -> R_fp8 - R_bf16 ~ +0.16; byte gap ~ -0.05
    (0.0625, 0.16 + 0.05, 1.0, "83-E2c b4 realization delta"),
    # b32: fp8 0.75x vs bf16 0.64x -> R_fp8 - R_bf16 ~ -0.29; byte gap ~ -0.25
    (0.500, -0.29 + 0.25, 1.0, "83-E2c b32 realization delta"),
]


def main() -> int:
    best = None
    for k0 in [x / 100 for x in range(0, 201, 2)]:
        for m0 in [x / 10 for x in range(2, 41, 1)]:
            err = sum(w * (k0 * max(0.0, 1 - m / m0) - k) ** 2
                      for m, k, w, _ in ROWS)
            if best is None or err < best[0]:
                best = (err, k0, m0)
    _, k0, m0 = best
    lines = [f"\n## kappa(M) fit: kappa = {k0:.2f} * max(0, 1 - M/{m0:.1f})",
             "| M | implied kappa | model | source |", "|---|---|---|---|"]
    for m, k, w, src in ROWS:
        lines.append(f"| {m:.3f} | {k:+.2f} | "
                     f"{k0 * max(0.0, 1 - m / m0):+.2f} | {src} |")
    lines.append(
        f"\n- FIT BASIS (honest): the paired realization deltas (same cell, "
        f"same beta, kernel-only difference) carry the fit; serve rows are "
        f"context (weights 0-0.2; the MLA b4/2k implied 0.97 is an S4-class "
        f"outlier -- its b8 neighbor implies ~0). Re-prediction: the flip "
        f"inversion (fp8 beats bf16-partial at M>=0.5, loses at M<=0.06) is "
        f"reproduced by construction of the ramp.\n"
        f"- Scope: fp8-dequant grouped-GEMM kernels; MoE Marlin shows the "
        f"same sign with smaller k0 (not separately fitted -- data within "
        f"noise). Selector rule: any fp8 draft realization prices with "
        f"kappa(M) at its CHAIN M, which is batch- and parallelism-"
        f"dependent -- constants alone CANNOT price a realization.")
    out = "\n".join(lines)
    with (PHASE / "results_exec.md").open("a") as f:
        f.write(out + "\n")
    (PHASE / "data/kappa_fit.json").write_text(
        json.dumps(dict(k0=k0, M0=m0), indent=1))
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
