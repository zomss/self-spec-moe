#!/usr/bin/env python3
"""P1 re-derived on MATCHED realizations.

Two comparisons, reported separately because they answer different
questions:

  DEPLOYMENT: every lever in its best AVAILABLE realization -- window
  singles get FULLCG (the engine only allows that chain with a
  window), quant/skip singles keep the plain chain because FULLCG is
  unavailable to them. Answers: "should I compose in production?"

  MECHANISM (pending plain-chain compositions): everything on the
  plain chain, so composition is isolated from the chain realization.
  Answers: "does composing cost terms help?"

Window singles: prefers the FULLCG control (cells_94ctl_*) when present,
else falls back to C1's plain measurement (flagged in the output).
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
        ctl = load(C2 / f"cells_94ctl_{arch}_*.csv", f"cells_94ctl_{arch}_")
        comps = load(C2 / f"cells_94_{arch}_*.csv", f"cells_94_{arch}_")
        upgraded = set()
        # replace plain window singles with their FULLCG control where we have it
        for cell, arms in ctl.items():
            for (n, K), v in arms.items():
                w = n.replace("fullcg", "")          # win512fullcg -> win512
                if (w, K) in singles.get(cell, {}):
                    singles[cell][(w, K)] = v
                    upgraded.add(w)
                else:
                    singles[cell][(w, K)] = v
        rows = []
        for cell in sorted(set(singles) & set(comps)):
            off = singles[cell].get(("off", 0))
            if not off:
                continue
            bs = max(((t / off[0], n, k) for (n, k), (t, a) in singles[cell].items()
                      if n != "off"), default=None)
            bc = max(((t / off[0], n, k) for (n, k), (t, a) in comps[cell].items()),
                     default=None)
            if not (bs and bc):
                continue
            gain = (bc[0] / bs[0] - 1) * 100
            comp_has_window = "win" in bc[1]
            single_is_window = bs[1].startswith("win")
            rows.append({"cell": f"b{cell[0]}/c{cell[1]}",
                         "best_single": f"{bs[1]}-K{bs[2]}", "S_single": round(bs[0], 3),
                         "best_comp": f"{bc[1]}-K{bc[2]}", "S_comp": round(bc[0], 3),
                         "gain_pct": round(gain, 1),
                         "matched_chain": bool(comp_has_window == single_is_window)})
        n5 = sum(1 for r in rows if r["gain_pct"] >= 5)
        npos = sum(1 for r in rows if r["gain_pct"] > 0)
        n5m = sum(1 for r in rows if r["gain_pct"] >= 5 and r["matched_chain"])
        nm = sum(1 for r in rows if r["matched_chain"])
        report[arch] = {"rows": rows, "n": len(rows), "n_ge5": n5, "n_pos": npos,
                        "n_matched": nm, "n_ge5_matched": n5m,
                        "window_singles_upgraded": sorted(upgraded)}
        print(f"\n===== {arch} (FULLCG window singles: {sorted(upgraded) or 'NONE'}) =====")
        for r in rows:
            flag = "" if r["matched_chain"] else "  [chain-mismatched]"
            print(f"{r['cell']:12s} single {r['best_single']:20s} {r['S_single']:.2f}"
                  f" | comp {r['best_comp']:26s} {r['S_comp']:.2f}"
                  f" | {r['gain_pct']:+6.1f}%{flag}")
        print(f"[{arch}] >=+5%: {n5}/{len(rows)}  (chain-matched subset: {n5m}/{nm})")

    OUT.mkdir(exist_ok=True)
    (OUT / "c2_p1_matched.json").write_text(json.dumps(report, indent=1))
    print("\nwrote", OUT / "c2_p1_matched.json")


if __name__ == "__main__":
    main()
