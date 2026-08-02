#!/usr/bin/env python3
"""Oracle analysis: demonstration 1 (composition beats single over a
COMPLETE space) + validation of the composed-from-singles prediction.

Config names: q-<quant>_w-<window>_s-<skip>, quant/window/skip each
"none" or a value. Exactly-one-active = SINGLE; >=2 active =
COMPOSITION; all-none = the bf16 self-draft (R_0 reference).
"""
import csv
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

DATA = Path("/data/smcho/self-spec-moe/research/94_composition/data")
OUT = Path("/data/smcho/self-spec-moe/paper/data")


def parse(name):
    d = {}
    for part in name.split("_"):
        k, _, v = part.partition("-")
        d[k] = v
    return d.get("q", "none"), d.get("w", "none"), d.get("s", "none")


def n_active(name):
    return sum(1 for x in parse(name) if x != "none")


C1DATA = Path("/data/smcho/self-spec-moe/research/93_c1_grid/data")


def load_ar(arch):
    """AR (off) decode_toks per cell, from C1 -- same protocol/machine."""
    ar = {}
    for f in glob.glob(str(C1DATA / f"cells_93_{arch}_off.csv")):
        for r in csv.DictReader(open(f)):
            ar[(int(r["batch"]), int(r["ctx"]))] = float(r["decode_toks"])
    return ar


def load(arch):
    cells = defaultdict(dict)
    for f in glob.glob(str(DATA / f"oracle_{arch}_*.csv")):
        name = Path(f).stem.split(f"oracle_{arch}_")[1]
        for r in csv.DictReader(open(f)):
            cells[(int(r["batch"]), int(r["ctx"]))][(name, int(r["K"]))] = (
                float(r["decode_toks"]), float(r["accept"] or 0))
    return cells


def main():
    for arch in (sys.argv[1:] or ["dense", "llama"]):
        cells = load(arch)
        ar = load_ar(arch)
        if not cells:
            print(f"[{arch}] no oracle data yet")
            continue
        base_name = "q-none_w-none_s-none"
        rows, pred_err = [], []
        print(f"\n===== {arch} oracle ({len(cells)} cells, "
              f"{len({n for c in cells.values() for n, _ in c})} configs) =====")
        for cell in sorted(cells):
            arms = cells[cell]
            # AR reference: the compile protocol's own off row is not in the
            # oracle, so use the bf16 self-draft (all-none) as the S=1 anchor
            if cell not in ar:
                continue
            base = (ar[cell], 0.0)   # AR anchor: S = toks / AR_toks
            singles = [(t / base[0], n, K) for (n, K), (t, a) in arms.items()
                       if n_active(n) == 1]
            comps = [(t / base[0], n, K) for (n, K), (t, a) in arms.items()
                     if n_active(n) >= 2]
            if not (singles and comps):
                continue
            bs, bc = max(singles), max(comps)
            gain = (bc[0] / bs[0] - 1) * 100
            rows.append({"cell": f"b{cell[0]}/c{cell[1]}",
                         "best_single": f"{bs[1]}-K{bs[2]}", "S_single": round(bs[0], 3),
                         "best_comp": f"{bc[1]}-K{bc[2]}", "S_comp": round(bc[0], 3),
                         "gain_pct": round(gain, 1)})
            # composed-from-singles acceptance prediction
            for (n, K), (t, a) in arms.items():
                if n_active(n) < 2 or a <= 1:
                    continue
                q, w, s = parse(n)
                fs = []
                for dim, val in (("q", q), ("w", w), ("s", s)):
                    if val == "none":
                        continue
                    sname = "_".join(f"{d}-{v if d == dim else 'none'}"
                                     for d, v in (("q", q), ("w", w), ("s", s)))
                    sv = arms.get((sname, K))
                    if sv and sv[1] > 1:
                        fs.append((sv[1] - 1) / K)
                if len(fs) < 2:
                    continue
                f_meas = (a - 1) / K
                prod = 1.0
                for x in fs:
                    prod *= x
                pred_err.append({"n_levers": len(fs),
                                 "ratio": f_meas / prod})
        for r in rows:
            m = "**" if r["gain_pct"] >= 5 else ("+" if r["gain_pct"] > 0 else " ")
            print(f"{r['cell']:12s} single {r['best_single']:26s} {r['S_single']:.2f}"
                  f" | comp {r['best_comp']:30s} {r['S_comp']:.2f}"
                  f" | {r['gain_pct']:+6.1f}% {m}")
        n5 = sum(1 for r in rows if r["gain_pct"] >= 5)
        print(f"[{arch}] composition >= +5% in {n5}/{len(rows)} cells "
              f"(complete factorial)")
        for nl in (2, 3):
            v = sorted(x["ratio"] for x in pred_err if x["n_levers"] == nl)
            if not v:
                continue
            n = len(v)
            print(f"[{arch}] composed f / product ({nl} levers): median "
                  f"{v[n//2]:.3f}  p10 {v[n//10]:.3f}  p90 {v[9*n//10]:.3f}"
                  f"  (n={n})")
        OUT.mkdir(exist_ok=True)
        (OUT / f"c2_oracle_{arch}.json").write_text(json.dumps(
            {"cells": rows, "n_ge5": n5,
             "composed_over_product": pred_err}, indent=1))
        print(f"wrote {OUT}/c2_oracle_{arch}.json")


if __name__ == "__main__":
    main()
