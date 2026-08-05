#!/usr/bin/env python3
"""Per-DEPTH acceptance profile extraction (Stage B of the C2 search).

Why per-depth: with a scalar f, dS/dK = (f - R)/(KR+1)^2 is
sign-constant, so optimal K would always be K_max or OFF -- no interior
optimum, contradicting every measured cell. f decays with depth
(measured -4% to -58% from K2 to K6 depending on lever), which is what
creates the interior argmax.

From tau(K) measured at several K we recover the mean per-position
acceptance in each depth band:

    tau(K) = 1 + sum_{i=1..K} p_i     =>     p_i band = (tau(K2)-tau(K1))/(K2-K1)

One profile (measured once at K_max) then yields tau(K) for EVERY
K <= K_max, and -- since acceptance is cell-independent (measured CV
2-3% across batch x ctx) -- for every cell. That is the amortization
that makes Stage B affordable.

Reads compile-cell CSVs (C1 singles, C2 compositions, oracle).
"""
import csv
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/data/smcho/self-spec-moe/research")
OUT = Path("/data/smcho/self-spec-moe/paper/data")


def collect(sources):
    """-> {arm: {K: [tau,...]}} pooled over cells (acceptance is
    cell-independent; pooling reduces noise). Later sources OVERRIDE
    earlier ones at (arm, K, batch, ctx) granularity -- used to overlay
    the post-collision-fix re-measurements (cells_93v2_*, see
    paper/data/c1_corruption_ledger.md) on the corrupt v1 rows."""
    rows = {}
    for pat, tag in sources:
        for f in glob.glob(str(pat)):
            name = Path(f).stem.split(tag)[1].replace("fullcg", "")
            name = name.removesuffix("_nc")
            for r in csv.DictReader(open(f)):
                tau = float(r["accept"] or 0)
                if tau > 1:
                    rows[(name, int(r["K"]), r["batch"], r["ctx"])] = tau
    per = defaultdict(lambda: defaultdict(list))
    for (name, K, _, _), tau in rows.items():
        per[name][K].append(tau)
    return per


def profile(taus):
    """{K: mean tau} -> per-depth band acceptance + cell-independence CV."""
    ks = sorted(taus)
    mean = {k: sum(v) / len(v) for k, v in taus.items()}
    cv = {k: (sum((x - mean[k]) ** 2 for x in v) / len(v)) ** 0.5 / mean[k]
          for k, v in taus.items() if len(v) > 1}
    bands, prev_k, prev_tau = [], 0, 1.0
    for k in ks:
        p = (mean[k] - prev_tau) / (k - prev_k)
        bands.append({"depth_from": prev_k + 1, "depth_to": k,
                      "p_mean": round(p, 4)})
        prev_k, prev_tau = k, mean[k]
    return {"tau_by_K": {str(k): round(mean[k], 4) for k in ks},
            "cell_CV_by_K": {str(k): round(cv.get(k, 0), 4) for k in ks},
            "depth_bands": bands,
            "decay_pct": (round((bands[-1]["p_mean"] / bands[0]["p_mean"] - 1) * 100, 1)
                          if len(bands) > 1 and bands[0]["p_mean"] else None),
            "n_cells": {str(k): len(taus[k]) for k in ks}}


def tau_at(prof, K):
    """Reconstruct tau(K) for any K <= K_max from the depth bands."""
    t, done = 1.0, 0
    for b in prof["depth_bands"]:
        take = max(0, min(b["depth_to"], K) - done)
        t += take * b["p_mean"]
        done += take
        if done >= K:
            break
    return t


def main():
    archs = sys.argv[1:] or ["dense", "llama", "q3_32b", "moe", "mla"]
    out = {}
    for arch in archs:
        oracle = ("llamafix" if arch == "llama" else arch)  # post-fix llama
        per = collect([
            (ROOT / f"93_c1_grid/data/cells_93_{arch}_*.csv", f"cells_93_{arch}_"),
            (ROOT / f"93_c1_grid/data/cells_93v2_{arch}_*.csv", f"cells_93v2_{arch}_"),
            (ROOT / f"94_composition/data/cells_94_{arch}_*.csv", f"cells_94_{arch}_"),
            (ROOT / f"94_composition/data/oracle_{oracle}_*.csv", f"oracle_{oracle}_"),
        ])
        if not per:
            continue
        profs = {a: profile(t) for a, t in per.items() if len(t) >= 2}
        out[arch] = profs
        print(f"\n===== {arch} =====")
        print(f"{'arm':28s} {'tau(K2)':>8s} {'tau(K4)':>8s} {'tau(K6)':>8s} "
              f"{'p_early':>8s} {'p_late':>8s} {'decay':>7s} {'cellCV':>7s}")
        for a, p in sorted(profs.items()):
            tb = p["tau_by_K"]
            b0 = p["depth_bands"][0]["p_mean"]
            bl = p["depth_bands"][-1]["p_mean"]
            cv = max(p["cell_CV_by_K"].values()) if p["cell_CV_by_K"] else 0
            print(f"{a:28s} {tb.get('2','-'):>8} {tb.get('4','-'):>8} "
                  f"{tb.get('6','-'):>8} {b0:8.3f} {bl:8.3f} "
                  f"{str(p['decay_pct'])+'%':>7s} {cv*100:6.1f}%")
        # reconstruction check: predict tau(K4) from bands built on K2,K6
        errs = []
        for a, p in profs.items():
            if "4" in p["tau_by_K"]:
                errs.append(abs(tau_at(p, 4) / float(p["tau_by_K"]["4"]) - 1))
        if errs:
            print(f"  tau(K) reconstruction |err|: mean {100*sum(errs)/len(errs):.2f}%")

    OUT.mkdir(exist_ok=True)
    (OUT / "c2_depth_profiles.json").write_text(json.dumps(out, indent=1))
    print("\nwrote", OUT / "c2_depth_profiles.json")


if __name__ == "__main__":
    main()
