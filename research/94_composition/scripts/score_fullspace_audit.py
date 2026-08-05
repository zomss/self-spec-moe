#!/usr/bin/env python3
"""Score the full-space sampled audit (fixed-seed uniform sample,
c2_fullspace_audit_plan.json) against the Step-3b search's picks.

Converts "regret vs the deployable space is a scaling extrapolation"
into a measured bound: for each cell, how many of the N uniformly
sampled (config, K) points beat the search's confirmed pick, and by
how much. With k=0 of N better, at 95% confidence at most
1 - 0.05^(1/N) of the full space beats the pick; with k>0 the
empirical fraction and the max exceedance are the bound.

Sample values come from fs_dense_* (audit + confirmation boots), the
C1 singles WITH the cells_93v2_* collision-repair overlay, or the
reduced-oracle files -- never from corrupt v1 rows.
"""
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path

C1 = Path("/data/smcho/self-spec-moe/research/93_c1_grid/data")
C2 = Path("/data/smcho/self-spec-moe/research/94_composition/data")
OUT = Path("/data/smcho/self-spec-moe/paper/data")

QUANTS = {"w4a16", "w4a8cut", "w4a8hum", "w8int8", "w8fp8", "fp8dyn"}
WINDOWS = {"win128", "win512", "win2048", "win8192"}


def read_rows(f, K):
    out = {}
    try:
        for r in csv.DictReader(open(f)):
            if r["arm"] == f"k{K}":
                out[(int(r["batch"]), int(r["ctx"]))] = float(r["decode_toks"])
    except OSError:
        pass
    return out


def sample_toks(name, K):
    """decode_toks per cell for one sampled point, corruption-safe."""
    fs = read_rows(C2 / f"fs_dense_{name.replace('+', '_')}.csv", K)
    if fs:
        return fs, "fs"
    if "+" not in name and name != "base":
        v1 = read_rows(C1 / f"cells_93_dense_{name}.csv", K)
        v2 = read_rows(C1 / f"cells_93v2_dense_{name}.csv", K)
        v2n = read_rows(C1 / f"cells_93v2_dense_{name}_nc.csv", K)
        if v1 or v2 or v2n:
            v1.update(v2)
            v1.update(v2n)
            return v1, "c1+v2overlay"
    parts = [] if name == "base" else name.split("+")
    q = next((p for p in parts if p in QUANTS), "none")
    w = next((p[3:] for p in parts if p in WINDOWS), "none")
    s = "b2" if "skipb2" in parts else ("b4" if "skipb4" in parts else "none")
    oq = {"none": "none", "w4a8hum": "hum", "w4a16": "w4a16"}.get(q)
    if oq and s != "b4" and w in ("none", "512", "2048"):
        o = read_rows(C2 / f"oracle_dense_q-{oq}_w-{w}_s-{s}.csv", K)
        if o:
            return o, "oracle"
    return {}, "MISSING"


def main():
    plan = json.load(open(OUT / "c2_fullspace_audit_plan.json"))
    result = json.load(open(OUT / "c2_fullspace_result.json"))["cells"]
    picks = {r["cell"]: r["S_full"] for r in result}
    ar = {}
    for r in csv.DictReader(open(C1 / "cells_93_dense_off.csv")):
        ar[(int(r["batch"]), int(r["ctx"]))] = float(r["decode_toks"])

    per_cell = defaultdict(list)   # cell -> [(S, name, K)]
    missing = []
    for name, K in plan["sample"]:
        toks, src = sample_toks(name, K)
        if not toks:
            missing.append(f"{name}-K{K}")
            continue
        for cell, t in toks.items():
            if cell in ar:
                per_cell[f"b{cell[0]}/c{cell[1]}"].append(
                    (t / ar[cell], name, K))

    rows = []
    print(f"{'cell':12s} {'n':>3s} {'pick S':>7s} {'beat':>5s} "
          f"{'max exceed':>11s} {'p95 bound':>9s}  best sampled")
    for cell in sorted(picks, key=lambda c: (int(c.split("/c")[1]),
                                             int(c[1:c.index("/")]))):
        samples = per_cell.get(cell, [])
        n = len(samples)
        pick = picks[cell]
        better = [(S, nm, K) for S, nm, K in samples if S > pick]
        exceed = (max(better)[0] / pick - 1) * 100 if better else 0.0
        p95 = (1 - 0.05 ** (1 / n)) * 100 if n and not better else \
              (len(better) / n * 100 if n else None)
        best = max(samples) if samples else None
        print(f"{cell:12s} {n:3d} {pick:7.3f} {len(better):5d} "
              f"{exceed:+10.2f}% {p95:8.1f}%  "
              f"{best[1]}-K{best[2]} S={best[0]:.3f}" if best else cell)
        rows.append({"cell": cell, "n_samples": n, "pick_S": pick,
                     "n_better": len(better),
                     "max_exceed_pct": round(exceed, 2),
                     "bound_pct_at_95": round(p95, 2) if p95 is not None else None,
                     "best_sampled": f"{best[1]}-K{best[2]}" if best else None,
                     "best_sampled_S": round(best[0], 3) if best else None})
    mean_exceed = sum(r["max_exceed_pct"] for r in rows) / len(rows)
    print(f"\nmean max-exceedance over cells: {mean_exceed:+.2f}%"
          f"   unmeasured sample points: {missing or 'none'}")
    (OUT / "c2_fullspace_audit.json").write_text(json.dumps(
        {"seed": plan["seed"], "n_sample": plan["n_sample"],
         "cells": rows, "mean_max_exceed_pct": round(mean_exceed, 2),
         "unmeasured": missing}, indent=1))
    print(f"wrote {OUT}/c2_fullspace_audit.json")


if __name__ == "__main__":
    main()
