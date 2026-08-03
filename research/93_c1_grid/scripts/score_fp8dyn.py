#!/usr/bin/env python3
"""Compare C1's fp8dyn column against the post-fix re-measurement.

C1's fp8dyn ran with TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 and, post-fix,
that configuration cannot compile at all -- so the C1 column was produced
by loading a foreign cached graph (w4a8cut's, per the shared key
7cd76306f0). This scores the honest re-measure against it, and against
w4a8cut, to see whose behaviour the old column actually reflected.
"""
import csv
import json
from pathlib import Path

C1 = Path("/data/smcho/self-spec-moe/research/93_c1_grid/data")
OUT = Path("/data/smcho/self-spec-moe/paper/data")


def load(f):
    d = {}
    for r in csv.DictReader(open(f)):
        d[(int(r["batch"]), int(r["ctx"]), int(r["K"]))] = (
            float(r["accept"] or 0), float(r["decode_toks"]))
    return d


def main():
    ar = {}
    for r in csv.DictReader(open(C1 / "cells_93_dense_off.csv")):
        ar[(int(r["batch"]), int(r["ctx"]))] = float(r["decode_toks"])
    old = load(C1 / "cells_93_dense_fp8dyn.csv")
    new = load(C1 / "cells_93v2_dense_fp8dyn_nc.csv")
    cut = load(C1 / "cells_93v2_dense_w4a8cut.csv")

    ks = sorted(set(old) & set(new))
    rows, dS, dtau, sim_old, sim_new = [], [], [], [], []
    for k in ks:
        if k[:2] not in ar:
            continue
        So, Sn = old[k][1] / ar[k[:2]], new[k][1] / ar[k[:2]]
        rows.append({"cell": f"b{k[0]}/c{k[1]}", "K": k[2],
                     "tau_old": round(old[k][0], 3), "tau_new": round(new[k][0], 3),
                     "S_old": round(So, 3), "S_new": round(Sn, 3),
                     "dS_pct": round(100 * (Sn / So - 1), 1)})
        dS.append(100 * (Sn / So - 1))
        dtau.append(100 * (new[k][0] / old[k][0] - 1) if old[k][0] else 0)
        if k in cut:                    # resemblance to the collision partner
            sim_old.append(abs(old[k][0] - cut[k][0]))
            sim_new.append(abs(new[k][0] - cut[k][0]))

    n = len(dS)
    print(f"dense fp8dyn: C1 (foreign graph) vs post-fix re-measure, n={n} cells\n")
    print(f"  {'cell':12s} {'K':>2s} {'tau old':>8s} {'tau new':>8s} "
          f"{'S old':>7s} {'S new':>7s} {'dS':>7s}")
    for r in rows:
        if r["K"] == 2 and r["cell"].split("/")[0] in ("b1", "b8", "b32"):
            print(f"  {r['cell']:12s} {r['K']:2d} {r['tau_old']:8.3f} "
                  f"{r['tau_new']:8.3f} {r['S_old']:7.3f} {r['S_new']:7.3f} "
                  f"{r['dS_pct']:+6.1f}%")
    md = sum(dS) / n
    print(f"\n  mean dS {md:+.1f}%   mean |dS| {sum(map(abs,dS))/n:.1f}%   "
          f"mean |dtau| {sum(map(abs,dtau))/n:.1f}%")
    if sim_old and sim_new:
        print(f"  |tau - w4a8cut|: OLD {sum(sim_old)/len(sim_old):.3f}  "
              f"NEW {sum(sim_new)/len(sim_new):.3f}   "
              f"(old column resembling w4a8cut is the collision signature)")
    won_o = sum(1 for r in rows if r["S_old"] > 1)
    won_n = sum(1 for r in rows if r["S_new"] > 1)
    print(f"  cells with S>1: {won_o} -> {won_n} of {n}")
    OUT.mkdir(exist_ok=True)
    (OUT / "c1_fp8dyn_corrected.json").write_text(json.dumps(
        {"cells": rows, "mean_dS_pct": round(md, 2),
         "mean_abs_dtau_pct": round(sum(map(abs, dtau)) / n, 2),
         "tau_dist_to_w4a8cut_old": round(sum(sim_old) / len(sim_old), 3) if sim_old else None,
         "tau_dist_to_w4a8cut_new": round(sum(sim_new) / len(sim_new), 3) if sim_new else None,
         "cells_S_gt1_old": won_o, "cells_S_gt1_new": won_n}, indent=1))
    print(f"\nwrote {OUT}/c1_fp8dyn_corrected.json")


if __name__ == "__main__":
    main()
