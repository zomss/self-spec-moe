#!/usr/bin/env python3
"""Phase 94 Step 1 analysis: P1 (composition beats best single) and
P2 (sub-additive beta) against the C1 single-lever cells.

Both sets were measured with the SAME compile protocol/stack/machine,
so decode_toks are directly comparable per (batch, ctx) cell.
"""
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path

C1 = Path("/data/smcho/self-spec-moe/research/93_c1_grid/data")
C2 = Path("/data/smcho/self-spec-moe/research/94_composition/data")
OUT = Path("/data/smcho/self-spec-moe/paper/data")


def load(pattern, tag):
    """-> {(batch,ctx): {(name,K): (toks, accept)}}"""
    cells = defaultdict(dict)
    for f in glob.glob(str(pattern)):
        name = Path(f).stem.split(tag)[1]
        for r in csv.DictReader(open(f)):
            cells[(int(r["batch"]), int(r["ctx"]))][(name, int(r["K"]))] = (
                float(r["decode_toks"]), float(r["accept"] or 0))
    return cells


def main():
    report = {}
    for arch in ("dense", "llama", "q3_32b"):
        singles = load(C1 / f"cells_93_{arch}_*.csv", f"cells_93_{arch}_")
        comps = load(C2 / f"cells_94_{arch}_*.csv", f"cells_94_{arch}_")
        rows = []
        for cell in sorted(set(singles) & set(comps)):
            off = singles[cell].get(("off", 0))
            if not off:
                continue
            best_s = max(((t / off[0], n, k, a)
                          for (n, k), (t, a) in singles[cell].items()
                          if n != "off"), default=None)
            best_c = max(((t / off[0], n, k, a)
                          for (n, k), (t, a) in comps[cell].items()),
                         default=None)
            if not (best_s and best_c):
                continue
            gain = (best_c[0] / best_s[0] - 1) * 100
            rows.append({
                "cell": f"b{cell[0]}/c{cell[1]}",
                "best_single": f"{best_s[1]}-K{best_s[2]}",
                "S_single": round(best_s[0], 3),
                "best_comp": f"{best_c[1]}-K{best_c[2]}",
                "S_comp": round(best_c[0], 3),
                "gain_pct": round(gain, 1),
                "accept_single": best_s[3], "accept_comp": best_c[3]})
        n5 = sum(1 for r in rows if r["gain_pct"] >= 5)
        npos = sum(1 for r in rows if r["gain_pct"] > 0)
        report[arch] = {"cells": rows, "n_cells": len(rows),
                        "n_gain_ge5pct": n5, "n_gain_pos": npos}
        print(f"\n===== {arch} =====")
        for r in rows:
            mark = "**" if r["gain_pct"] >= 5 else ("+" if r["gain_pct"] > 0 else " ")
            print(f"{r['cell']:12s} single {r['best_single']:22s} {r['S_single']:.2f}"
                  f" | comp {r['best_comp']:26s} {r['S_comp']:.2f}"
                  f" | {r['gain_pct']:+6.1f}% {mark}")
        print(f"[{arch}] composition >= +5% in {n5}/{len(rows)} cells; "
              f"any gain in {npos}/{len(rows)}")

    # P2: sub-additivity check on accept (composed vs product of singles)
    OUT.mkdir(exist_ok=True)
    (OUT / "c2_premise_probe.json").write_text(json.dumps(report, indent=1))
    print("\nwrote", OUT / "c2_premise_probe.json")

    tot5 = sum(r["n_gain_ge5pct"] for r in report.values())
    tot = sum(r["n_cells"] for r in report.values())
    archs_pass = sum(1 for a, r in report.items()
                     if r["n_cells"] and r["n_gain_ge5pct"] >= r["n_cells"] / 2)
    print(f"\nP1 (>=5% at >=half cells on >=2 arch): {archs_pass} arch pass; "
          f"overall {tot5}/{tot} cells -> "
          f"{'CONFIRMED' if archs_pass >= 2 else 'REFUTED'}")


if __name__ == "__main__":
    main()
