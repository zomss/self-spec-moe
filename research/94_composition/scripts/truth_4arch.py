#!/usr/bin/env python3
"""Selection rules on FOUR architectures, judged against the 2-sample truth.

The 4-arch ranker table (c2_mechanism_4arch.json) was scored against
single-sample argmax truths, which carry winner's-curse inflation
measured at 3.9-6.3pp -- larger than the 0.3-0.5pp margins separating
the rules on MLA/MoE. With the R2 replications (independent content,
doc offset 16) every arch gets a 2-sample truth: per-(config,K) mean S
over both samples, argmax over the means.

Each rule picks per cell from ONE sample's measured singles (the
measurements a real search would have made), confirms top-5 within
that sample, and is scored by the 2-SAMPLE mean S of what it chose.
Both directions (pick-from-R1, pick-from-R2) are reported; a rule that
only wins in one direction is winning on content luck.

Sample pairs: dense (oracle_, oracleR2_), llama (oracle_llamafix_ +
oracle_llamaR2_ -- R1 is collision-corrupt, see c2.md correction),
mla/moe (oracle_, oracleR2_).
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

PAIRS = {
    "dense": ("oracle_dense_", "oracleR2_dense_"),
    "llama": ("oracle_llamafix_", "oracle_llamaR2_"),
    "mla":   ("oracle_mla_",   "oracleR2_mla_"),
    "moe":   ("oracle_moe_",   "oracleR2_moe_"),
}
ANCHOR = {"llama": "cells_93_llama_off.csv"}   # default cells_93_<arch>_off


def load(prefix):
    cells = defaultdict(dict)
    for f in glob.glob(str(C2 / (prefix + "*.csv"))):
        n = Path(f).stem.split(prefix)[1]
        for r in csv.DictReader(open(f)):
            cells[(int(r["batch"]), int(r["ctx"]))][(n, int(r["K"]))] = (
                float(r["decode_toks"]), float(r["accept"] or 0))
    return cells


def main():
    rules = bl.make_scorers()
    report = {}
    for arch in (sys.argv[1:] or list(PAIRS)):
        p1, p2 = PAIRS[arch]
        s1, s2 = load(p1), load(p2)
        if not s1 or not s2:
            print(f"[{arch}] missing sample ({p1}: {bool(s1)}, {p2}: {bool(s2)})")
            continue
        ar = {}
        for r in csv.DictReader(open(C1 / ANCHOR.get(arch, f"cells_93_{arch}_off.csv"))):
            ar[(int(r["batch"]), int(r["ctx"]))] = float(r["decode_toks"])
        cells = sorted(c for c in set(s1) & set(s2) if c in ar)
        Sm, truth = {}, {}
        for c in cells:
            ks = set(s1[c]) & set(s2[c])
            Sm[c] = {k: (s1[c][k][0] + s2[c][k][0]) / (2 * ar[c]) for k in ks}
            truth[c] = max(Sm[c], key=Sm[c].get)

        def reg(picks):
            rs = [(Sm[c][truth[c]] - Sm[c].get((picks[c][1], picks[c][2]), 0))
                  / Sm[c][truth[c]] * 100 for c in cells if c in picks]
            return sum(rs) / len(rs) if rs else None

        rows = {}
        for name in ("ours", "knapspec", "product_rank"):
            r1 = reg(bl.rank_and_pick({c: s1[c] for c in cells}, ar, rules[name], 5))
            r2 = reg(bl.rank_and_pick({c: s2[c] for c in cells}, ar, rules[name], 5))
            rows[name] = {"from_R1": round(r1, 3), "from_R2": round(r2, 3),
                          "mean": round((r1 + r2) / 2, 3)}
        for tag, s in (("exhaustive_1sample_R1", s1), ("exhaustive_1sample_R2", s2)):
            picks = {c: (0,) + max(s[c], key=lambda k: s[c][k][0]) for c in cells}
            picks = {c: (0, k[1], k[2]) for c, k in picks.items()}
            rows[tag] = round(reg(picks), 3)
        n_moved = sum(1 for c in cells
                      if max(s1[c], key=lambda k: s1[c][k][0])
                      != max(s2[c], key=lambda k: s2[c][k][0]))
        rows["argmax_moved_cells"] = f"{n_moved}/{len(cells)}"
        rows["best_rule_mean"] = min(
            (n for n in ("ours", "knapspec", "product_rank")),
            key=lambda n: rows[n]["mean"])
        report[arch] = rows
        print(f"\n=== {arch} ({len(cells)} cells, 2-sample truth) ===")
        for n in ("ours", "knapspec", "product_rank"):
            r = rows[n]
            print(f"  {n:14s} from-R1 {r['from_R1']:6.2f}%  "
                  f"from-R2 {r['from_R2']:6.2f}%  mean {r['mean']:6.2f}%")
        print(f"  exhaustive-1-sample: R1 {rows['exhaustive_1sample_R1']:.2f}%  "
              f"R2 {rows['exhaustive_1sample_R2']:.2f}%   "
              f"argmax moved {rows['argmax_moved_cells']}  "
              f"best rule: {rows['best_rule_mean']}")
    OUT.mkdir(exist_ok=True)
    (OUT / "c2_truth_4arch.json").write_text(json.dumps(report, indent=1))
    print(f"\nwrote {OUT}/c2_truth_4arch.json")


if __name__ == "__main__":
    main()
