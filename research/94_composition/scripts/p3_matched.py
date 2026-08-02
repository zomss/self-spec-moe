#!/usr/bin/env python3
"""P3 (cost additivity) — BASELINE-FREE formulation.

The naive test needs R_0 (bf16 self-draft, no lever), which was never
measured. This avoids it: if cost terms are additive, the cost effect
of adding lever B must be INDEPENDENT of which lever A it is added to.

  delta_B|A  =  R(A x B) - R(A)

Additivity  <=>  delta_B|A1 == delta_B|A2 for different A.
Also tested: delta_B|A vs delta_B|(A x C)  (independence given a third
lever) -- the triple-composition check.

R inverted from measured cells: R = ((accept/S) - 1)/K.
Realization note: window-containing configs use FULLCG, others plain;
the measured chain effect is -0.5% mean / +-4% (see ctl analysis), so
deltas carry roughly that much noise.
"""
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path

C1 = Path("/data/smcho/self-spec-moe/research/93_c1_grid/data")
C2 = Path("/data/smcho/self-spec-moe/research/94_composition/data")
OUT = Path("/data/smcho/self-spec-moe/paper/data")

# (arch, added lever B, [(base A, composed A x B), ...])
TESTS = [
    ("dense", "win512", [("w4a8hum", "hum_x_win512"), ("w4a16", "w4a16_x_win512")]),
    ("dense", "skipb2", [("w4a8hum", "hum_x_skipb2"),
                         ("win512", "win512_x_skipb2")]),
    ("dense", "skipb2|given win512", [("hum_x_win512", "hum_x_win512_x_skipb2")]),
    ("llama", "win512", [("w4a16", "w4a16_x_win512")]),
    ("llama", "win2048", [("w4a16", "w4a16_x_win2048")]),
    ("llama", "skipb2", [("w4a16", "w4a16_x_skipb2")]),
    ("q3_32b", "win512", [("w4gptq", "w4gptq_x_win512")]),
    ("q3_32b", "win2048", [("w4gptq", "w4gptq_x_win2048")]),
]


def load_all(arch):
    cells = defaultdict(dict)
    for pat, tag in ((C1 / f"cells_93_{arch}_*.csv", f"cells_93_{arch}_"),
                     (C2 / f"cells_94_{arch}_*.csv", f"cells_94_{arch}_"),
                     (C2 / f"cells_94ctl_{arch}_*.csv", f"cells_94ctl_{arch}_")):
        for f in glob.glob(str(pat)):
            name = Path(f).stem.split(tag)[1].replace("fullcg", "")
            for r in csv.DictReader(open(f)):
                cells[(int(r["batch"]), int(r["ctx"]))][(name, int(r["K"]))] = (
                    float(r["decode_toks"]), float(r["accept"] or 0))
    return cells


def R_of(cells, cell, name, K):
    off = cells[cell].get(("off", 0))
    v = cells[cell].get((name, K))
    if not (off and v) or v[1] <= 1:
        return None
    S = v[0] / off[0]
    if S <= 0:
        return None
    return ((v[1] / S) - 1.0) / K


def main():
    out = []
    print("BASELINE-FREE ADDITIVITY: is the cost effect of lever B the "
          "same wherever it is added?\n")
    for arch, lever, pairs in TESTS:
        cells = load_all(arch)
        per_cell = defaultdict(list)
        for cell in sorted(cells):
            for base, comp in pairs:
                for K in (2, 4, 6):
                    Rb = R_of(cells, cell, base, K)
                    Rc = R_of(cells, cell, comp, K)
                    if Rb is None or Rc is None:
                        continue
                    per_cell[(cell, K)].append((base, Rc - Rb))
        # cells where >=2 different bases measured the same lever
        spreads = []
        for (cell, K), ds in per_cell.items():
            if len(ds) < 2:
                continue
            vals = [d for _, d in ds]
            spread = max(vals) - min(vals)
            scale = max(abs(v) for v in vals) or 1e-6
            spreads.append((spread, spread / scale, cell, K, ds))
        if not spreads:
            # single-base lever: report the delta distribution instead
            allv = [d for ds in per_cell.values() for _, d in ds]
            if allv:
                print(f"{arch:7s} +{lever:20s} (single base) delta R: "
                      f"mean {sum(allv)/len(allv):+.3f}  "
                      f"range [{min(allv):+.3f}, {max(allv):+.3f}]  n={len(allv)}")
                out.append({"arch": arch, "lever": lever, "mode": "single_base",
                            "delta_mean": sum(allv) / len(allv),
                            "delta_min": min(allv), "delta_max": max(allv),
                            "n": len(allv)})
            continue
        rel = [s[1] for s in spreads]
        rel.sort()
        med = rel[len(rel) // 2]
        print(f"{arch:7s} +{lever:20s} bases differ by: median {med*100:5.1f}% "
              f"of the effect  (n={len(spreads)} cells)")
        for spread, r, cell, K, ds in sorted(spreads, key=lambda x: -x[1])[:2]:
            detail = ", ".join(f"{b}:{d:+.3f}" for b, d in ds)
            print(f"          worst b{cell[0]}/c{cell[1]} K{K}: {detail}"
                  f"  (spread {spread:+.3f}, {r*100:.0f}%)")
        out.append({"arch": arch, "lever": lever, "mode": "multi_base",
                    "median_rel_spread": med, "n_cells": len(spreads)})

    OUT.mkdir(exist_ok=True)
    (OUT / "c2_p3_matched.json").write_text(json.dumps(out, indent=1))
    print("\nwrote", OUT / "c2_p3_matched.json")
    print("\nREADING: small relative spread => the lever's cost effect is "
          "base-independent => additive nomination is sound.\n"
          "Large spread => cost terms INTERACT; Stage-1 pruning needs a "
          "measured correction.")


if __name__ == "__main__":
    main()
