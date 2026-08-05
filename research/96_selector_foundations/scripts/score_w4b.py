#!/usr/bin/env python3
"""W4b scorer: audited (notune) llama cells vs the lottery-era oracle.

R inversion per (config, K, batch, ctx), identical to compile_from_c2:
    tau   = accept
    S_ref = decode_toks / decode_toks_off(batch, ctx)
    R     = (tau / S_ref - 1) / K

Old (lottery-era): mean of oracle_llamafix_ / oracle_llamaR2_ replications
(what compile_from_c2 --sample mean fed the policy tables), anchored on
cells_93_llama_off. New: w4audit_* CSVs, notune, own off anchor.

Output: per-cell R_old vs R_new and the inflation ratio; uniformity check
against the pre-registered 1.46x (results_w2.md F6).
"""
import csv
from pathlib import Path

ROOT = Path("/data/smcho/self-spec-moe")
OLD = ROOT / "research/94_composition/data"
OLD_ANCHOR = ROOT / "research/93_c1_grid/data/cells_93_llama_off.csv"
NEW = ROOT / "research/96_selector_foundations/data/w4"

CFGS = ["q-w4a16_w-512_s-b2", "q-w4a16_w-2048_s-b2"]


def read_cells(path):
    """(K, batch, ctx) -> (decode_toks, accept); off rows key K=0."""
    out = {}
    if not path.exists():
        return out
    for r in csv.DictReader(open(path)):
        out[(int(r["K"]), int(r["batch"]), int(r["ctx"]))] = (
            float(r["decode_toks"]), float(r["accept"]))
    return out


def rmap(cells, anchor):
    """(K, b, ctx) -> R via the compile_from_c2 inversion."""
    out = {}
    for (k, b, c), (toks, acc) in cells.items():
        if k == 0:
            continue
        off = anchor.get((0, b, c))
        if not off or off[0] <= 0 or toks <= 0:
            continue
        s = toks / off[0]
        out[(k, b, c)] = ((acc / s) - 1) / k
    return out


def mean_maps(maps):
    keys = set().union(*(m.keys() for m in maps if m))
    out = {}
    for k in keys:
        vs = [m[k] for m in maps if m and k in m]
        if vs:
            out[k] = sum(vs) / len(vs)
    return out


old_anchor = read_cells(OLD_ANCHOR)
new_anchor = read_cells(NEW / "w4audit_llama_off.csv")

print(f"{'config':22s} {'K':>2s} {'b':>3s} {'ctx':>6s} | "
      f"{'R_old':>7s} {'R_new':>7s} {'infl':>6s} | "
      f"{'acc_old':>7s} {'acc_new':>7s}")
ratios = []
for cfg in CFGS:
    olds = [rmap(read_cells(OLD / f"{p}{cfg}.csv"), old_anchor)
            for p in ("oracle_llamafix_", "oracle_llamaR2_")]
    r_old = mean_maps(olds)
    a_old = mean_maps([{k: v[1] for k, v in
                        read_cells(OLD / f"{p}{cfg}.csv").items() if k[0]}
                       for p in ("oracle_llamafix_", "oracle_llamaR2_")])
    new_cells = read_cells(NEW / f"w4audit_llama_{cfg}.csv")
    r_new = rmap(new_cells, new_anchor)
    for key in sorted(set(r_old) & set(r_new)):
        k, b, c = key
        infl = r_old[key] / r_new[key] if r_new[key] > 0 else float("nan")
        ratios.append((cfg, key, infl))
        print(f"{cfg:22s} {k:>2d} {b:>3d} {c:>6d} | "
              f"{r_old[key]:7.4f} {r_new[key]:7.4f} {infl:6.2f} | "
              f"{a_old.get(key, float('nan')):7.3f} "
              f"{new_cells[key][1]:7.3f}")

if ratios:
    vals = [x for _, _, x in ratios if x == x]
    mean = sum(vals) / len(vals)
    lo, hi = min(vals), max(vals)
    spread = (hi - lo) / mean
    print(f"\ninflation: mean {mean:.2f}x  range [{lo:.2f}, {hi:.2f}]  "
          f"spread {spread:.1%}")
    print(f"pre-registered F6 prediction: 1.46x uniform.")
    print("UNIFORM (rescale the table)" if spread < 0.20 else
          "NON-UNIFORM (regenerate the llama table)")
