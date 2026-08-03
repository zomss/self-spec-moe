#!/usr/bin/env python3
"""The honest headline: our search vs EXHAUSTIVE, both judged against
the 2-sample truth (mean of two independent content samples).

Also asks the deployment question that actually matters: config
IDENTITY churns because the top configs are within ~5% of each other,
but is the achieved SPEEDUP stable?
"""
import csv
import glob
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

P = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("bl", P / "baselines.py")
bl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bl)

C1 = Path("/data/smcho/self-spec-moe/research/93_c1_grid/data")
C2 = Path("/data/smcho/self-spec-moe/research/94_composition/data")
OUT = Path("/data/smcho/self-spec-moe/paper/data")


def raw(arch, rep=""):
    cells = defaultdict(dict)
    for f in glob.glob(str(C2 / f"oracle{rep}_{arch}_*.csv")):
        n = Path(f).stem.split(f"oracle{rep}_{arch}_")[1]
        for r in csv.DictReader(open(f)):
            cells[(int(r["batch"]), int(r["ctx"]))][(n, int(r["K"]))] = (
                float(r["decode_toks"]), float(r["accept"] or 0))
    return cells


def main():
    rep = {}
    for arch in ("dense", "llama"):
        ar = {}
        for r in csv.DictReader(open(C1 / f"cells_93_{arch}_off.csv")):
            ar[(int(r["batch"]), int(r["ctx"]))] = float(r["decode_toks"])
        r1, r2 = raw(arch, ""), raw(arch, "R2")
        cells_c = sorted(set(r1) & set(r2))
        Sm = {}
        for c in cells_c:
            ks = sorted(set(r1[c]) & set(r2[c]))
            Sm[c] = {k: (r1[c][k][0] + r2[c][k][0]) / (2 * ar[c]) for k in ks}

        # our search, derived independently from each sample
        picks = {}
        for tag, data in (("R1", r1), ("R2", r2)):
            bl_cells = {c: data[c] for c in cells_c}
            picks[tag] = bl.rank_and_pick(bl_cells, ar, bl.make_scorers()["ours"], 5)

        print(f"\n=== {arch}: judged against the 2-sample truth")
        print(f"  {'cell':12s} {'true best':24s} {'ours(R1)':>9s} "
              f"{'ours(R2)':>9s} {'exh(R1)':>8s}  same pick?")
        rows = []
        for c in cells_c:
            best = max(Sm[c].values())
            truth = max(Sm[c], key=Sm[c].get)
            a1 = max(r1[c], key=lambda k: r1[c][k][0])
            def reg(k):
                return (best - Sm[c].get(k, 0)) / best * 100
            p1 = (picks["R1"][c][1], picks["R1"][c][2]) if c in picks["R1"] else None
            p2 = (picks["R2"][c][1], picks["R2"][c][2]) if c in picks["R2"] else None
            same = "yes" if p1 == p2 else "no"
            print(f"  b{c[0]}/c{c[1]:<6} {truth[0]+'-K'+str(truth[1]):24s} "
                  f"{reg(p1):8.2f}% {reg(p2):8.2f}% {reg(a1):7.2f}%  {same}")
            rows.append({"cell": f"b{c[0]}/c{c[1]}", "true_best": f"{truth[0]}-K{truth[1]}",
                         "regret_ours_R1_pct": round(reg(p1), 2),
                         "regret_ours_R2_pct": round(reg(p2), 2),
                         "regret_exhaustive_R1_pct": round(reg(a1), 2),
                         "same_pick_across_samples": same == "yes"})
        n = len(rows)
        mo1 = sum(r["regret_ours_R1_pct"] for r in rows) / n
        mo2 = sum(r["regret_ours_R2_pct"] for r in rows) / n
        me = sum(r["regret_exhaustive_R1_pct"] for r in rows) / n
        stab = sum(1 for r in rows if r["same_pick_across_samples"])
        print(f"  -> OURS   : {mo1:.2f}% (from R1) / {mo2:.2f}% (from R2)")
        print(f"  -> EXHAUSTIVE (all 36 configs, 1 sample): {me:.2f}%")
        print(f"  -> our pick identical across the two samples in {stab}/{n} cells")
        rep[arch] = {"cells": rows, "mean_ours_R1_pct": round(mo1, 2),
                     "mean_ours_R2_pct": round(mo2, 2),
                     "mean_exhaustive_R1_pct": round(me, 2),
                     "pick_stable_cells": stab, "n_cells": n}
    (OUT / "c2_search_vs_truth.json").write_text(json.dumps(rep, indent=1))
    print(f"\nwrote {OUT}/c2_search_vs_truth.json")


if __name__ == "__main__":
    main()
