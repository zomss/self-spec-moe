#!/usr/bin/env python3
"""Set-search backtest: prior-guided measurement allocation vs baselines.

Ground truth = the measured set pool (beta_menu_ext.csv: ls_g_/ls_na_/ls_p_
arms), grouped by budget (3/5/7 dropped layers). The search question at a
budget: find the pool-best set with as few MEASUREMENTS as possible (each
~45 s), given only the singles profile (24 leave-one-out betas) as prior.

Strategies:
  prior-guided  measure sets in descending product-prior order; after m
                measurements pick the best MEASURED set
  random        expected regret of measuring m sets uniformly (exact
                expectation over the pool)
  profile-solve pick the product-prior argmax with NO measurement
                (KnapSpec-style profile-then-solve analog) = prior-guided
                at m=0

Also reports prior quality per budget (Spearman rank corr, prior vs
measured) -- the depth-degradation of the product law, quantified.

Scope note: this validates the ALLOCATION rule over the measured pool, not
global optimality over 2^24 sets. Output: data/set_search_backtest.md
"""

import csv
import itertools
from pathlib import Path

PHASE = Path(__file__).resolve().parents[1]
CSV = PHASE / "data/beta_menu_ext.csv"


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def main() -> int:
    singles, sets = {}, {}
    for r in csv.DictReader(CSV.open()):
        arm, b = r["arm"], float(r["beta_greedy"])
        if arm.startswith("ls_") and arm.count("_") == 1:
            singles[int(arm[3:])] = b
        elif arm.startswith(("ls_g_", "ls_na_", "ls_p_")):
            ids = tuple(int(x) for x in arm.split("_")[-1].split("-"))
            sets[ids] = b

    lines = ["# Set-search backtest — prior-guided allocation over the measured pool",
             "", f"Pool: {len(sets)} measured sets; prior = product of the",
             "24 leave-one-out singles. Regret = pool-best beta minus chosen",
             "beta. profile-solve = prior argmax with zero measurements.", ""]
    for budget in (3, 5, 7):
        pool = {ids: b for ids, b in sets.items() if len(ids) == budget}
        if len(pool) < 3:
            continue
        prior = {ids: 1.0 for ids in pool}
        for ids in pool:
            for i in ids:
                prior[ids] *= singles.get(i, 1.0)
        best = max(pool.values())
        order = sorted(pool, key=lambda s: -prior[s])       # prior-guided
        rho = spearman([prior[s] for s in pool], [pool[s] for s in pool])
        lines.append(f"\n## budget {budget} — pool {len(pool)}, best beta "
                     f"{best:.3f}, prior-vs-measured Spearman rho={rho:.2f}\n")
        lines.append("| m measured | prior-guided regret | random regret (exp.) |")
        lines.append("|---|---|---|")
        n = len(pool)
        for m in range(0, n + 1):
            if m == 0:
                pg = best - pool[order[0]]                  # profile-solve
                rnd = best - sum(pool.values()) / n         # random single pick
            else:
                pg = best - max(pool[s] for s in order[:m])
                # E[max of m uniform draws]: exact over combinations
                vals = sorted(pool.values())
                exp_max = 0.0
                total = 0
                for comb in itertools.combinations(range(n), m):
                    exp_max += max(vals[i] for i in comb)
                    total += 1
                rnd = best - exp_max / total
            lines.append(f"| {m} | {pg:+.3f} | {rnd:+.3f} |")
            if pg == 0 and m > 0:
                lines.append(f"| … | prior-guided CONVERGED at m={m} "
                             f"({m}×45 s ≈ {m*45//60} min) | |")
                break
        # detail
        lines.append("\n| set | prior | measured |")
        lines.append("|---|---|---|")
        for s in order:
            lines.append(f"| {{{','.join(map(str, s))}}} | {prior[s]:.3f} "
                         f"| {pool[s]:.3f} |")
    text = "\n".join(lines)
    (PHASE / "data/set_search_backtest.md").write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
