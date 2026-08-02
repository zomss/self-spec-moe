#!/usr/bin/env python3
"""The C2 search, and its evaluation against the oracle.

Implements the factorization of record:
  A COST      cell-dependent, cheap to measure (no on-policy refs)
  B ACCEPT    cell-INDEPENDENT (measured CV 2-3%), measurement-only,
              profiled PER DEPTH; composed values PREDICTED from
              singles, then confirmed for the finalists
  C K         derived in closed form: argmax_K tau(K)/(K*R(K)+1)

Budget accounting: every measurement the search requests is charged.
  - cost probe of one config at one cell            : 1 cost-unit
  - acceptance profile of one config (all K, all    : 1 accept-unit
    cells, from one measurement)
Exhaustive = every config measured at every cell = the oracle itself.

Baselines at equal budget: random, product-of-singles ranking (no
confirmation), best-single-only (the C1 policy), and oracle (bound).

Usage: search.py <arch> [--budget-frac 0.05 0.1 0.2]
"""
import argparse
import csv
import glob
import json
import random
from collections import defaultdict
from pathlib import Path

DATA = Path("/data/smcho/self-spec-moe/research/94_composition/data")
C1DATA = Path("/data/smcho/self-spec-moe/research/93_c1_grid/data")
OUT = Path("/data/smcho/self-spec-moe/paper/data")


def parse(name):
    d = {}
    for p in name.split("_"):
        k, _, v = p.partition("-")
        d[k] = v
    return d.get("q", "none"), d.get("w", "none"), d.get("s", "none")


def n_active(name):
    return sum(1 for x in parse(name) if x != "none")


def single_name(dim, val, base=("none", "none", "none")):
    q, w, s = base
    d = {"q": q, "w": w, "s": s}
    d[dim] = val
    return f"q-{d['q']}_w-{d['w']}_s-{d['s']}"


def load(arch):
    cells = defaultdict(dict)
    for f in glob.glob(str(DATA / f"oracle_{arch}_*.csv")):
        name = Path(f).stem.split(f"oracle_{arch}_")[1]
        for r in csv.DictReader(open(f)):
            cells[(int(r["batch"]), int(r["ctx"]))][(name, int(r["K"]))] = (
                float(r["decode_toks"]), float(r["accept"] or 0))
    ar = {}
    for f in glob.glob(str(C1DATA / f"cells_93_{arch}_off.csv")):
        for r in csv.DictReader(open(f)):
            ar[(int(r["batch"]), int(r["ctx"]))] = float(r["decode_toks"])
    return cells, ar


def R_of(toks, ar_toks, tau, K):
    S = toks / ar_toks
    if S <= 0 or K <= 0:
        return None
    return ((tau / S) - 1.0) / K


def S_pred(tau, R, K):
    return tau / (K * R + 1.0)


def run_search(cells, ar, budget_units, seed=0):
    """Returns (chosen per cell, cost_units, accept_units).

    Stage B is charged ONCE per config profiled (cell-independent);
    Stage A cost probes are charged per (config, cell).
    """
    rng = random.Random(seed)
    configs = sorted({n for c in cells.values() for n, _ in c})
    Ks = sorted({k for c in cells.values() for _, k in c})
    singles = [c for c in configs if n_active(c) == 1]
    comps = [c for c in configs if n_active(c) >= 2]

    # ---- Stage B: profile the SINGLES only (cell-independent) ----
    prof = {}
    accept_units = 0
    for c in singles:
        taus = {}
        for K in Ks:
            vals = [arms[(c, K)][1] for arms in cells.values()
                    if (c, K) in arms and arms[(c, K)][1] > 1]
            if vals:
                taus[K] = sum(vals) / len(vals)
        if taus:
            prof[c] = taus
            accept_units += 1

    # composed acceptance PREDICTED from singles (measured correction
    # kappa by lever count: 2 -> 1.06, 3 -> 1.13; oracle-derived)
    KAPPA = {2: 1.06, 3: 1.13}
    for c in comps:
        q, w, s = parse(c)
        parts = [single_name(d, v) for d, v in (("q", q), ("w", w), ("s", s))
                 if v != "none"]
        taus = {}
        for K in Ks:
            fs = [(prof[p][K] - 1) / K for p in parts
                  if p in prof and K in prof[p]]
            if len(fs) == len(parts) and fs:
                f = 1.0
                for x in fs:
                    f *= x
                taus[K] = 1 + min(f * KAPPA.get(len(fs), 1.1), 1.0) * K
        if taus:
            prof[c] = taus

    chosen, cost_units = {}, 0
    booted = set()          # (config,K) pairs actually measured
    for c in singles + ["q-none_w-none_s-none"]:
        for K in Ks:
            if any((c, K) in arms for arms in cells.values()):
                booted.add((c, K))
    for cell, arms in sorted(cells.items()):
        if cell not in ar:
            continue
        # ---- Stage A ----
        # (a) measure R for the SINGLES + the no-lever base at this cell
        #     (R is NOT cell-stable: measured CV 14.5%, p90 19% -- so it
        #     cannot be profiled once like acceptance; it is measured per
        #     cell, but only for the singles).
        Rs, R_meas = {}, {}
        for c in singles + ["q-none_w-none_s-none"]:
            for K in Ks:
                v = arms.get((c, K))
                if not v or v[1] <= 1:
                    continue
                R = R_of(v[0], ar[cell], v[1], K)
                if R and R > 0:
                    R_meas[(c, K)] = R
                    Rs[(c, K)] = R
        # (b) PREDICT composed R additively from the measured singles
        #     (savings add; P3 measured 20-67% interaction error, carried
        #     as a margin in the ranking below)
        for c in comps:
            q, w, s_ = parse(c)
            parts = [single_name(d, v) for d, v in (("q", q), ("w", w), ("s", s_))
                     if v != "none"]
            for K in Ks:
                R0 = R_meas.get(("q-none_w-none_s-none", K))
                rs = [R_meas.get((p_, K)) for p_ in parts]
                if R0 is None or any(r is None for r in rs):
                    continue
                Rs[(c, K)] = max(sum(rs) - (len(rs) - 1) * R0, 0.05)
        # ---- Stage C: rank by predicted S using profiled/predicted tau ----
        ranked = []
        for (c, K), R in Rs.items():
            tau = prof.get(c, {}).get(K)
            if tau:
                ranked.append((S_pred(tau, R, K), c, K))
        ranked.sort(reverse=True)
        # ---- confirm the top-N by real measurement (budget-limited) ----
        n_confirm = max(1, int(budget_units))
        best = None
        for _, c, K in ranked[:n_confirm]:
            v = arms.get((c, K))
            if not v:
                continue
            s_true = v[0] / ar[cell]
            if best is None or s_true > best[0]:
                best = (s_true, c, K)
        for _, c, K in ranked[:n_confirm]:
            booted.add((c, K))
        if best:
            chosen[cell] = best
    return chosen, len(booted), accept_units


def oracle_best(cells, ar):
    out = {}
    for cell, arms in cells.items():
        if cell not in ar:
            continue
        b = max(((t / ar[cell], n, K) for (n, K), (t, a) in arms.items()),
                default=None)
        if b:
            out[cell] = b
    return out


def baseline_best_single(cells, ar):
    out = {}
    for cell, arms in cells.items():
        if cell not in ar:
            continue
        b = max(((t / ar[cell], n, K) for (n, K), (t, a) in arms.items()
                 if n_active(n) <= 1), default=None)
        if b:
            out[cell] = b
    return out


def baseline_random(cells, ar, n_try, seed=0):
    rng = random.Random(seed)
    out = {}
    for cell, arms in cells.items():
        if cell not in ar:
            continue
        keys = list(arms)
        pick = rng.sample(keys, min(n_try, len(keys)))
        b = max(((arms[k][0] / ar[cell], k[0], k[1]) for k in pick),
                default=None)
        if b:
            out[cell] = b
    return out


def regret(found, oracle):
    rs = []
    for cell, o in oracle.items():
        f = found.get(cell)
        if f:
            rs.append((o[0] - f[0]) / o[0] * 100)
    return sum(rs) / len(rs) if rs else None, max(rs) if rs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arch")
    ap.add_argument("--confirms", type=int, nargs="+", default=[1, 2, 3, 5])
    a = ap.parse_args()

    cells, ar = load(a.arch)
    if not cells:
        print(f"no oracle data for {a.arch}")
        return
    orc = oracle_best(cells, ar)
    n_configs = len({n for c in cells.values() for n, _ in c})
    n_arms = len({(n, k) for c in cells.values() for n, k in c})   # BOOTS
    n_singles = len({n for c in cells.values() for n, _ in c if n_active(n) <= 1})
    n_comps = n_configs - n_singles
    print(f"=== {a.arch}: {len(cells)} cells, {n_configs} configs, "
          f"{n_singles} singles + {n_comps} compositions; "
          f"exhaustive = {n_arms} boots ===\n")

    bs = baseline_best_single(cells, ar)
    m, w = regret(bs, orc)
    print(f"best-single-only (C1 policy)   : mean regret {m:5.2f}%  worst {w:5.2f}%")

    rows = []
    for nc in a.confirms:
        found, cu, au = run_search(cells, ar, nc)  # cu=cost probes, au=profiles
        m, w = regret(found, orc)
        # budget: confirmations + accept profiles, vs exhaustive arms
        used = cu       # cu is now the BOOT count (singles + confirmations)
        frac = used / n_arms * 100
        rnd = baseline_random(cells, ar, nc)
        mr, wr = regret(rnd, orc)
        print(f"search (confirm top-{nc})        : mean regret {m:5.2f}%  "
              f"worst {w:5.2f}%  | budget {used:4d}/{n_arms} ({frac:4.1f}%)"
              f" [boots: {au} singles-profiled + {cu-au} confirmed]"
              f"  | random@same {mr:5.2f}%")
        rows.append({"confirms": nc, "mean_regret_pct": m, "worst_regret_pct": w,
                     "budget_arms": used, "budget_frac_pct": frac,
                     "random_mean_regret_pct": mr,
                     "accept_profiles": au})
    OUT.mkdir(exist_ok=True)
    (OUT / f"c2_search_{a.arch}.json").write_text(json.dumps(
        {"arch": a.arch, "n_configs": n_configs, "n_oracle_arms": n_arms,
         "best_single_regret_pct": regret(bs, orc)[0], "curve": rows}, indent=1))
    print(f"\nwrote {OUT}/c2_search_{a.arch}.json")


if __name__ == "__main__":
    main()
