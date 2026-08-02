#!/usr/bin/env python3
"""C2 Step 3b: run the search on the FULL lever space.

The point of the factorization: singles are measured ONCE and reused,
so enlarging the composition space costs only the confirmations. Here
the singles come free -- C1 Stage A already measured all 13 of them
with the same protocol/stack/machine.

Full dense space: quant{none,6} x window{none,4} x skip{none,2}
= 105 lever-combos x K{2,3,4,6} = 420 (config,K) exhaustive boots.

Phase 1 (this script, --plan): predict every combo from the reused
singles, emit the top-N per cell as the confirmation list.
Phase 2 (--score, after measuring): compare the search's picks against
the reduced-oracle optimum to show the larger space was exploited.
"""
import argparse
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path

C1 = Path("/data/smcho/self-spec-moe/research/93_c1_grid/data")
C2 = Path("/data/smcho/self-spec-moe/research/94_composition/data")
OUT = Path("/data/smcho/self-spec-moe/paper/data")
KAPPA = {1: 1.0, 2: 1.06, 3: 1.13}

QUANTS = ["w4a16", "w4a8cut", "w4a8hum", "w8int8", "w8fp8", "fp8dyn"]
WINDOWS = ["win128", "win512", "win2048", "win8192"]
SKIPS = ["skipb2", "skipb4"]
KS = [2, 3, 4, 6]


def load_singles(arch="dense"):
    cells = defaultdict(dict)
    ar = {}
    for f in glob.glob(str(C1 / f"cells_93_{arch}_*.csv")):
        n = Path(f).stem.split(f"cells_93_{arch}_")[1]
        for r in csv.DictReader(open(f)):
            cell = (int(r["batch"]), int(r["ctx"]))
            if n == "off":
                ar[cell] = float(r["decode_toks"])
            else:
                cells[cell][(n, int(r["K"]))] = (
                    float(r["decode_toks"]), float(r["accept"] or 0))
    return cells, ar


def depth_bands(taus):
    """{K: tau} -> per-depth band acceptance, for tau(K) at any K."""
    bands, pk, pt = [], 0, 1.0
    for k in sorted(taus):
        bands.append((pk + 1, k, (taus[k] - pt) / (k - pk)))
        pk, pt = k, taus[k]
    return bands


def tau_at(bands, K):
    t, done = 1.0, 0
    for lo, hi, p in bands:
        take = max(0, min(hi, K) - done)
        t += take * p
        done += take
        if done >= K:
            break
    if done < K and bands:                     # extrapolate the last band
        t += (K - done) * bands[-1][2]
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=5)
    a = ap.parse_args()

    cells, ar = load_singles()
    # per-single depth bands (acceptance is cell-independent: pool cells)
    prof = {}
    for name in QUANTS + WINDOWS + SKIPS:
        taus = {}
        for K in (2, 4, 6):
            v = [arms[(name, K)][1] for arms in cells.values()
                 if (name, K) in arms and arms[(name, K)][1] > 1]
            if v:
                taus[K] = sum(v) / len(v)
        if taus:
            prof[name] = depth_bands(taus)

    plan = defaultdict(set)
    per_cell = {}
    for cell, arms in sorted(cells.items()):
        if cell not in ar:
            continue
        # measured single R at this cell, per K (interpolate across K)
        Rs = {}
        for name in QUANTS + WINDOWS + SKIPS:
            for K in (2, 4, 6):
                v = arms.get((name, K))
                if v and v[1] > 1:
                    S = v[0] / ar[cell]
                    if S > 0:
                        Rs[(name, K)] = ((v[1] / S) - 1.0) / K
        def R_at(name, K):
            if (name, K) in Rs:
                return Rs[(name, K)]
            near = [(abs(k - K), Rs[(name, k)]) for k in (2, 4, 6)
                    if (name, k) in Rs]
            return min(near)[1] if near else None

        ranked = []
        for q in [None] + QUANTS:
            for w in [None] + WINDOWS:
                for s in [None] + SKIPS:
                    parts = [x for x in (q, w, s) if x]
                    if not parts:
                        continue
                    for K in KS:
                        rs = [R_at(p, K) for p in parts]
                        if any(r is None for r in rs):
                            continue
                        # additive cost (savings add), R0 ~ 1.0 for a
                        # bf16 self-draft
                        R = max(sum(rs) - (len(rs) - 1) * 1.0, 0.05)
                        fs = []
                        for p in parts:
                            if p not in prof:
                                break
                            fs.append((tau_at(prof[p], K) - 1) / K)
                        if len(fs) != len(parts):
                            continue
                        f = 1.0
                        for x in fs:
                            f *= x
                        f = min(f * KAPPA.get(len(parts), 1.15), 1.0)
                        S = (1 + f * K) / (K * R + 1)
                        ranked.append((S, "+".join(parts), K))
        ranked.sort(reverse=True)
        per_cell[f"b{cell[0]}/c{cell[1]}"] = [
            {"config": n, "K": K, "S_pred": round(s, 3)}
            for s, n, K in ranked[:a.top]]
        for s, n, K in ranked[:a.top]:
            plan[n].add(K)

    n_combos = (len(QUANTS) + 1) * (len(WINDOWS) + 1) * (len(SKIPS) + 1) - 1
    exhaustive = n_combos * len(KS)
    n_singles = len(QUANTS) + len(WINDOWS) + len(SKIPS)
    n_confirm = sum(len(v) for v in plan.values())
    print(f"FULL SPACE: {n_combos} lever-combos x {len(KS)} K = "
          f"{exhaustive} (config,K) boots exhaustive")
    print(f"  singles REUSED from C1 (already paid): {n_singles} configs")
    print(f"  new confirmations required           : {n_confirm} boots")
    print(f"  marginal cost                        : "
          f"{100*n_confirm/exhaustive:.1f}% of exhaustive")
    print(f"  (incl. singles as if unpaid          : "
          f"{100*(n_confirm+n_singles*3)/exhaustive:.1f}%)")
    print("\nconfirmation list:")
    for n, ks in sorted(plan.items()):
        print(f"  {n:34s} K{sorted(ks)}")
    OUT.mkdir(exist_ok=True)
    (OUT / "c2_fullspace_plan.json").write_text(json.dumps(
        {"n_combos": n_combos, "exhaustive_boots": exhaustive,
         "singles_reused": n_singles, "confirm_boots": n_confirm,
         "marginal_pct": 100 * n_confirm / exhaustive,
         "per_cell_topN": per_cell,
         "plan": {k: sorted(v) for k, v in plan.items()}}, indent=1))
    print(f"\nwrote {OUT}/c2_fullspace_plan.json")


if __name__ == "__main__":
    main()
