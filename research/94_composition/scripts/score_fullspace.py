#!/usr/bin/env python3
"""C2 Step 3b scoring: did searching the FULL space actually pay?

Compares, per cell:
  reduced-oracle best   -- the true optimum of the 18-config oracle
  full-space search     -- best of what the search chose to confirm
                           from the 105-combo space
A win here means the search reached configurations the reduced oracle
could not represent (extra quant formats, win128/win8192, skipb4,
K=3) and they measured better.
"""
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path

C1 = Path("/data/smcho/self-spec-moe/research/93_c1_grid/data")
C2 = Path("/data/smcho/self-spec-moe/research/94_composition/data")
P82 = Path("/data/smcho/self-spec-moe/research/82_runtime_switching/data")
OUT = Path("/data/smcho/self-spec-moe/paper/data")


def load(pattern, tag, sep="_"):
    cells = defaultdict(dict)
    for f in glob.glob(str(pattern)):
        name = Path(f).stem.split(tag)[1]
        for r in csv.DictReader(open(f)):
            cells[(int(r["batch"]), int(r["ctx"]))][(name, int(r["K"]))] = (
                float(r["decode_toks"]), float(r["accept"] or 0))
    return cells


def main():
    ar = {}
    for f in glob.glob(str(C1 / "cells_93_dense_off.csv")):
        for r in csv.DictReader(open(f)):
            ar[(int(r["batch"]), int(r["ctx"]))] = float(r["decode_toks"])
    oracle = load(C2 / "oracle_dense_*.csv", "oracle_dense_")
    fs = load(P82 / "fs_dense_*.csv", "fs_dense_")

    rows, wins = [], 0
    print(f"{'cell':12s} {'reduced-oracle best':34s} {'S':>5s} | "
          f"{'full-space search pick':34s} {'S':>5s} | gain")
    for cell in sorted(oracle):
        if cell not in ar:
            continue
        ob = max(((t / ar[cell], n, K) for (n, K), (t, a) in oracle[cell].items()),
                 default=None)
        fb = max(((t / ar[cell], n, K) for (n, K), (t, a) in fs.get(cell, {}).items()),
                 default=None)
        if not (ob and fb):
            continue
        gain = (fb[0] / ob[0] - 1) * 100
        if gain > 0:
            wins += 1
        rows.append({"cell": f"b{cell[0]}/c{cell[1]}",
                     "oracle_best": f"{ob[1]}-K{ob[2]}", "S_oracle": round(ob[0], 3),
                     "fullspace_pick": f"{fb[1]}-K{fb[2]}", "S_full": round(fb[0], 3),
                     "gain_pct": round(gain, 1)})
        m = "**" if gain >= 2 else ("+" if gain > 0 else " ")
        print(f"b{cell[0]}/c{cell[1]:<6} {ob[1][:34]:34s} {ob[0]:5.2f} | "
              f"{fb[1][:34]:34s} {fb[0]:5.2f} | {gain:+5.1f}% {m}")
    print(f"\nfull-space search beat the reduced-oracle optimum in "
          f"{wins}/{len(rows)} cells")
    mean = sum(r["gain_pct"] for r in rows) / len(rows) if rows else 0
    print(f"mean gain over the reduced-oracle optimum: {mean:+.1f}%")
    OUT.mkdir(exist_ok=True)
    (OUT / "c2_fullspace_result.json").write_text(json.dumps(
        {"cells": rows, "n_wins": wins, "mean_gain_pct": mean}, indent=1))
    print(f"wrote {OUT}/c2_fullspace_result.json")


if __name__ == "__main__":
    main()
