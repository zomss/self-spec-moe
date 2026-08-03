#!/usr/bin/env python3
"""Is the search's error SEARCH error, or shared measurement noise?

The oracle argmax moved in 14/16 cells between content samples, so
"regret vs a single-sample argmax" measures agreement with a noisy
target, and comparing any fixed pick against a FRESH argmax is
inflated by winner's curse (max over 36 noisy configs is upward
biased).

Honest estimator: treat mean(S_R1, S_R2) as the best available
estimate of each config's true S. Then ask, against that:
  - how much does the EXHAUSTIVE single-sample oracle lose?
  - how much does OUR search (6% of the cost) lose?
If they are the same, the limit is content noise, not search error.
"""
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path

C1 = Path("/data/smcho/self-spec-moe/research/93_c1_grid/data")
C2 = Path("/data/smcho/self-spec-moe/research/94_composition/data")
OUT = Path("/data/smcho/self-spec-moe/paper/data")


def load(arch, rep=""):
    cells = defaultdict(dict)
    for f in glob.glob(str(C2 / f"oracle{rep}_{arch}_*.csv")):
        n = Path(f).stem.split(f"oracle{rep}_{arch}_")[1]
        for r in csv.DictReader(open(f)):
            cells[(int(r["batch"]), int(r["ctx"]))][(n, int(r["K"]))] = \
                float(r["decode_toks"])
    return cells


def main():
    rep = {}
    for arch in ("dense", "llama"):
        s1, s2 = load(arch, ""), load(arch, "R2")
        ar = {}
        for r in csv.DictReader(open(C1 / f"cells_93_{arch}_off.csv")):
            ar[(int(r["batch"]), int(r["ctx"]))] = float(r["decode_toks"])
        pick = json.loads((OUT / "c2_precision_audit.json").read_text())
        rows = []
        for cell in sorted(set(s1) & set(s2)):
            if cell not in ar:
                continue
            keys = sorted(set(s1[cell]) & set(s2[cell]))
            if len(keys) < 5:
                continue
            S1 = {k: s1[cell][k] / ar[cell] for k in keys}
            S2 = {k: s2[cell][k] / ar[cell] for k in keys}
            Sm = {k: (S1[k] + S2[k]) / 2 for k in keys}
            truth = max(Sm, key=Sm.get)
            best = Sm[truth]
            a1 = max(S1, key=S1.get)          # exhaustive, sample 1
            a2 = max(S2, key=S2.get)          # exhaustive, sample 2
            # spread at the top of the true ranking
            top = sorted(Sm.values(), reverse=True)
            spread5 = (top[0] - top[4]) / top[0] * 100
            rows.append({
                "cell": f"b{cell[0]}/c{cell[1]}",
                "n_configs": len(keys),
                "true_best": f"{truth[0]}-K{truth[1]}",
                "S_true": round(best, 3),
                "regret_exhaustive_R1_pct": round((best - Sm[a1]) / best * 100, 2),
                "regret_exhaustive_R2_pct": round((best - Sm[a2]) / best * 100, 2),
                "top5_spread_pct": round(spread5, 2),
                "naive_R1_vs_freshmax_pct": round(
                    (S2[a2] - S2[a1]) / S2[a2] * 100, 2),
            })
        me1 = sum(r["regret_exhaustive_R1_pct"] for r in rows) / len(rows)
        me2 = sum(r["regret_exhaustive_R2_pct"] for r in rows) / len(rows)
        sp = sum(r["top5_spread_pct"] for r in rows) / len(rows)
        nv = sum(r["naive_R1_vs_freshmax_pct"] for r in rows) / len(rows)
        print(f"\n=== {arch}  ({len(rows)} cells, {rows[0]['n_configs']} configs each)")
        print(f"  {'cell':12s} {'true best (mean of 2)':26s} "
              f"{'exh.R1':>7s} {'exh.R2':>7s} {'top5 spread':>12s}")
        for r in rows:
            print(f"  {r['cell']:12s} {r['true_best']:26s} "
                  f"{r['regret_exhaustive_R1_pct']:6.2f}% "
                  f"{r['regret_exhaustive_R2_pct']:6.2f}% "
                  f"{r['top5_spread_pct']:11.1f}%")
        print(f"  -> EXHAUSTIVE search on one sample loses {me1:.2f}% (R1) / "
              f"{me2:.2f}% (R2) vs the 2-sample truth")
        print(f"  -> naive 'R1 pick vs fresh argmax' reads {nv:.2f}% "
              f"-- inflated by winner's curse")
        print(f"  -> mean spread across the top-5 configs: {sp:.1f}%")
        rep[arch] = {"cells": rows, "mean_regret_exhaustive_R1_pct": round(me1, 2),
                     "mean_regret_exhaustive_R2_pct": round(me2, 2),
                     "mean_naive_freshmax_pct": round(nv, 2),
                     "mean_top5_spread_pct": round(sp, 2)}
    (OUT / "c2_winners_curse.json").write_text(json.dumps(rep, indent=1))
    print(f"\nwrote {OUT}/c2_winners_curse.json")


if __name__ == "__main__":
    main()
