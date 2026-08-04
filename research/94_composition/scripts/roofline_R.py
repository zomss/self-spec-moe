#!/usr/bin/env python3
"""Test a ROOFLINE model for R against the log-linear surface.

A decode step is overhead-bound at tiny batch, memory-bound in the middle,
and compute-bound at large batch. Those transitions are PIECEWISE, which is
why a log-linear surface in (log batch, log ctx) leaves ~9% error against a
~5% noise floor. The roofline form encodes the transition explicitly:

    T(B, L, levers) = max(T_mem, T_compute) + T_overhead
    T_mem     = sigma * (a_W * theta + a_KV * B * L_eff)
    T_compute = sigma * a_C * B
    L_eff     = min(L, window + sinks)
    R         = T_draft / T_target

theta = weight-byte ratio vs bf16, sigma = surviving-layer fraction, both
known per lever. R is scale-invariant, so a_W is pinned to 1 and only
(a_KV, a_C, F) are fitted -- three parameters for the whole architecture.

Evaluation is leave-one-CELL-out: fit on the other cells, predict every
lever's R at the held-out cell. That is the operational question -- can
Stage A skip measuring a cell?
"""
import csv
import glob
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

C1 = Path("/data/smcho/self-spec-moe/research/93_c1_grid/data")
OUT = Path("/data/smcho/self-spec-moe/paper/data")

LAYERS = {"dense": 36, "llama": 32}
THETA = {"none": 1.0, "w4a16": 0.25, "w4a8cut": 0.25, "w4a8hum": 0.25,
         "w8int8": 0.5, "w8fp8": 0.5, "fp8dyn": 0.5, "w8chan": 0.5}
NSKIP = {"skipb2": 2, "skipb4": 4}


def lever_factors(arm, ctx, layers):
    """(theta, L_eff, sigma) for a single-lever arm name."""
    theta, leff, nskip = 1.0, ctx, 0
    if arm in THETA and arm != "none":
        theta = THETA[arm]
    if arm.startswith("win"):
        leff = min(ctx, int(arm[3:]) + 16)
    if arm in NSKIP:
        nskip = NSKIP[arm]
    return theta, leff, (layers - nskip) / layers


def load_R(arch):
    ar = {}
    for r in csv.DictReader(open(C1 / f"cells_93_{arch}_off.csv")):
        ar[(int(r["batch"]), int(r["ctx"]))] = float(r["decode_toks"])
    rows = []
    for f in sorted(glob.glob(str(C1 / f"cells_93_{arch}_*.csv"))):
        arm = Path(f).stem.split(f"cells_93_{arch}_")[1]
        if arm == "off" or arm == "kvq":
            continue
        v2 = C1 / f"cells_93v2_{arch}_{arm}.csv"
        src = v2 if v2.exists() else Path(f)
        for r in csv.DictReader(open(src)):
            cell = (int(r["batch"]), int(r["ctx"]))
            a = float(r["accept"] or 0)
            K = int(r["K"])
            if cell not in ar or a <= 1:
                continue
            S = float(r["decode_toks"]) / ar[cell]
            if S <= 0:
                continue
            R = ((a / S) - 1) / K
            if R and R > 0:
                rows.append({"cell": cell, "arm": arm, "K": K, "R": R})
    return rows


def predict(params, row, layers):
    a_kv, a_c, F = np.exp(params)          # positivity by construction
    B, L = row["cell"]
    th, leff, sg = lever_factors(row["arm"], L, layers)
    mem_t = 1.0 + a_kv * B * L
    com_t = a_c * B
    mem_d = sg * (1.0 * th + a_kv * B * leff)
    com_d = sg * a_c * B
    return (max(mem_d, com_d) + F) / (max(mem_t, com_t) + F)


def fit(rows, layers):
    def resid(p):
        return [(predict(p, r, layers) - r["R"]) / r["R"] for r in rows]
    best, bcost = None, np.inf
    for seed in ([-8, -6, -3], [-10, -8, -1], [-6, -9, -4], [-12, -5, 0]):
        try:
            s = least_squares(resid, seed, max_nfev=4000)
            if s.cost < bcost:
                bcost, best = s.cost, s.x
        except Exception:
            continue
    return best


def main():
    report = {}
    for arch, noise, loglin in (("dense", 5.6, 9.1), ("llama", 4.6, 9.8)):
        rows = load_R(arch)
        layers = LAYERS[arch]
        cells = sorted({r["cell"] for r in rows})
        errs = []
        for held in cells:
            tr = [r for r in rows if r["cell"] != held]
            te = [r for r in rows if r["cell"] == held]
            p = fit(tr, layers)
            if p is None:
                continue
            for r in te:
                errs.append(abs(predict(p, r, layers) - r["R"]) / r["R"] * 100)
        m, med = float(np.mean(errs)), float(np.median(errs))
        print(f"\n=== {arch}: leave-one-cell-out, {len(cells)} cells, n={len(errs)}")
        print(f"   roofline    mean {m:5.1f}%   median {med:5.1f}%")
        print(f"   log-linear  mean {loglin:5.1f}%   (previous model)")
        print(f"   noise floor      {noise:5.1f}%")
        verdict = ("BEATS log-linear" if m < loglin else "no better than log-linear")
        print(f"   -> roofline {verdict}"
              f"{'; AT the noise floor' if m <= noise else ''}")
        report[arch] = {"roofline_mean_pct": round(m, 2),
                        "roofline_median_pct": round(med, 2),
                        "loglinear_mean_pct": loglin, "noise_floor_pct": noise,
                        "n": len(errs)}
    OUT.mkdir(exist_ok=True)
    (OUT / "c2_roofline_R.json").write_text(json.dumps(report, indent=1))
    print(f"\nwrote {OUT}/c2_roofline_R.json")


if __name__ == "__main__":
    main()
