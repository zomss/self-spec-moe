#!/usr/bin/env python3
"""Independent replication: does the SEARCH make the same DECISION on a
fresh document sample?

Precision of a search strategy is not "the regret number is small"; it
is "re-run it on independent content and it chooses the same thing".
Compares run R1 (original documents) against R2 (doc offset 16):
  - oracle argmax per cell        -> is the GROUND TRUTH itself stable?
  - search pick per cell          -> is our DECISION stable?
  - regret                        -> does the headline reproduce?
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
            cells[(int(r["batch"]), int(r["ctx"]))][(n, int(r["K"]))] = (
                float(r["decode_toks"]), float(r["accept"] or 0))
    ar = {}
    for r in csv.DictReader(open(C1 / f"cells_93_{arch}_off.csv")):
        ar[(int(r["batch"]), int(r["ctx"]))] = float(r["decode_toks"])
    return cells, ar


def argmax(cells, ar):
    out = {}
    for cell, arms in cells.items():
        if cell not in ar:
            continue
        b = max(((t / ar[cell], n, K) for (n, K), (t, a) in arms.items()),
                default=None)
        if b:
            out[cell] = b
    return out


def main():
    rep = {}
    for arch in ("dense", "llama"):
        c1, ar = load(arch, "")
        c2, _ = load(arch, "R2")
        if not c2:
            print(f"{arch}: replication not present yet")
            continue
        a1, a2 = argmax(c1, ar), argmax(c2, ar)
        common = sorted(set(a1) & set(a2))
        same_truth = sum(1 for c in common
                         if (a1[c][1], a1[c][2]) == (a2[c][1], a2[c][2]))
        print(f"\n=== {arch}: R1 vs R2 (independent document sample), "
              f"{len(common)} cells")
        print(f"  {'cell':12s} {'R1 oracle argmax':26s} {'R2 oracle argmax':26s}")
        rows = []
        for c in common:
            n1 = f"{a1[c][1]}-K{a1[c][2]}"
            n2 = f"{a2[c][1]}-K{a2[c][2]}"
            flag = "" if n1 == n2 else "   <== ground truth moved"
            print(f"  b{c[0]}/c{c[1]:<6} {n1:26s} {n2:26s}{flag}")
            # what does the R1 winner score on R2 content?  (regret of
            # committing to the R1 decision, judged on fresh content)
            key = (a1[c][1], a1[c][2])
            s_r1_on_r2 = (c2[c][key][0] / ar[c]) if key in c2[c] else None
            reg = ((a2[c][0] - s_r1_on_r2) / a2[c][0] * 100
                   if s_r1_on_r2 else None)
            rows.append({"cell": f"b{c[0]}/c{c[1]}", "r1_argmax": n1,
                         "r2_argmax": n2, "same": n1 == n2,
                         "S_r2_best": round(a2[c][0], 3),
                         "S_r1pick_on_r2": round(s_r1_on_r2, 3) if s_r1_on_r2 else None,
                         "regret_of_r1_decision_pct": round(reg, 2) if reg is not None else None})
        regs = [r["regret_of_r1_decision_pct"] for r in rows
                if r["regret_of_r1_decision_pct"] is not None]
        mean = sum(regs) / len(regs) if regs else 0
        print(f"  -> ground truth identical in {same_truth}/{len(common)} cells")
        print(f"  -> committing to the R1 choice costs {mean:+.2f}% mean "
              f"(worst {max(regs):+.2f}%) judged on R2 content")
        rep[arch] = {"cells": rows, "same_argmax": same_truth,
                     "n_cells": len(common),
                     "mean_regret_of_r1_decision_pct": round(mean, 3),
                     "worst_pct": round(max(regs), 2) if regs else None}
    if rep:
        (OUT / "c2_replication_audit.json").write_text(json.dumps(rep, indent=1))
        print(f"\nwrote {OUT}/c2_replication_audit.json")


if __name__ == "__main__":
    main()
