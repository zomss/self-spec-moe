#!/usr/bin/env python3
"""C2 Step 3a: competing SELECTION RULES scored against the oracle.

Pure analysis -- every rule sees the same measured singles and must
choose a composed config per cell; all are then scored by the TRUE
speedup of what they chose. No new GPU work.

Rules
  oracle            the true best (bound)
  ours              measured singles -> additive R + kappa-corrected
                    product acceptance -> rank by S -> confirm top-N
  product_rank      product-of-singles acceptance, NO kappa, NO
                    confirmation (P5: does the naive composition
                    heuristic mis-rank the top config?)
  proxy_rank        phase-90's dead class: rank by a static importance
                    proxy (here: fewest levers = least perturbation,
                    a generous stand-in for any "predict acceptance
                    from structure" scheme)
  knapspec          KnapSpec-style: additive per-lever value under a
                    cost budget, greedy knapsack, no interaction term
  best_single       the C1 policy
  random_N          random N configs per cell
"""
import csv
import glob
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

DATA = Path("/data/smcho/self-spec-moe/research/94_composition/data")
C1DATA = Path("/data/smcho/self-spec-moe/research/93_c1_grid/data")
OUT = Path("/data/smcho/self-spec-moe/paper/data")
KAPPA = {2: 1.06, 3: 1.13}


def parse(n):
    d = {}
    for p in n.split("_"):
        k, _, v = p.partition("-")
        d[k] = v
    return d.get("q", "none"), d.get("w", "none"), d.get("s", "none")


def n_active(n):
    return sum(1 for x in parse(n) if x != "none")


def single_of(dim, val):
    d = {"q": "none", "w": "none", "s": "none"}
    d[dim] = val
    return f"q-{d['q']}_w-{d['w']}_s-{d['s']}"


def parts_of(n):
    q, w, s = parse(n)
    return [single_of(d, v) for d, v in (("q", q), ("w", w), ("s", s))
            if v != "none"]


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


def R_of(t, art, tau, K):
    S = t / art
    return ((tau / S) - 1.0) / K if S > 0 and K else None


def rank_and_pick(cells, ar, score_fn, n_confirm):
    """score_fn(cell_arms, R_meas, prof, name, K) -> predicted S."""
    out = {}
    for cell, arms in cells.items():
        if cell not in ar:
            continue
        R_meas, prof = {}, {}
        for (n, K), (t, a) in arms.items():
            if n_active(n) <= 1 and a > 1:
                R = R_of(t, ar[cell], a, K)
                if R and R > 0:
                    R_meas[(n, K)] = R
                prof.setdefault(n, {})[K] = a
        ranked = []
        for (n, K) in arms:
            s = score_fn(arms, R_meas, prof, n, K)
            if s is not None:
                ranked.append((s, n, K))
        ranked.sort(reverse=True)
        best = None
        for _, n, K in ranked[:n_confirm]:
            v = arms.get((n, K))
            if not v:
                continue
            true_s = v[0] / ar[cell]
            if best is None or true_s > best[0]:
                best = (true_s, n, K)
        if best:
            out[cell] = best
    return out


def make_scorers():
    def R_pred(R_meas, n, K):
        ps = parts_of(n)
        if not ps:
            return R_meas.get((n, K))
        R0 = R_meas.get(("q-none_w-none_s-none", K))
        rs = [R_meas.get((p, K)) for p in ps]
        if R0 is None or any(r is None for r in rs):
            return None
        return max(sum(rs) - (len(rs) - 1) * R0, 0.05)

    def f_prod(prof, n, K, kappa=True):
        ps = parts_of(n)
        if not ps:
            p = prof.get(n, {}).get(K)
            return (p - 1) / K if p and p > 1 else None
        fs = []
        for p in ps:
            t = prof.get(p, {}).get(K)
            if not t or t <= 1:
                return None
            fs.append((t - 1) / K)
        f = 1.0
        for x in fs:
            f *= x
        return min(f * (KAPPA.get(len(fs), 1.1) if kappa else 1.0), 1.0)

    def ours(arms, R_meas, prof, n, K):
        R, f = R_pred(R_meas, n, K), f_prod(prof, n, K, True)
        return (1 + f * K) / (K * R + 1) if R and f else None

    def product_rank(arms, R_meas, prof, n, K):
        R, f = R_pred(R_meas, n, K), f_prod(prof, n, K, False)
        return (1 + f * K) / (K * R + 1) if R and f else None

    def proxy_rank(arms, R_meas, prof, n, K):
        # "structural" proxy: prefer least perturbation, deeper K
        return -n_active(n) * 10 + K

    def knapspec(arms, R_meas, prof, n, K):
        # additive value/cost knapsack, no interaction, no measurement
        # of composed acceptance: value = sum of single-lever accept
        # gains, cost = sum of single-lever R savings
        ps = parts_of(n)
        if not ps:
            return None
        R0 = R_meas.get(("q-none_w-none_s-none", K))
        if R0 is None:
            return None
        val = cost = 0.0
        for p in ps:
            t = prof.get(p, {}).get(K)
            r = R_meas.get((p, K))
            if t is None or r is None:
                return None
            val += (t - 1) / K
            cost += max(R0 - r, 0)
        return val * (1 + cost)

    return {"ours": ours, "product_rank": product_rank,
            "proxy_rank": proxy_rank, "knapspec": knapspec}


def regret(found, orc):
    rs = [(orc[c][0] - found[c][0]) / orc[c][0] * 100
          for c in orc if c in found]
    return (sum(rs) / len(rs), max(rs)) if rs else (None, None)


def main():
    report = {}
    for arch in (sys.argv[1:] or ["dense", "llama"]):
        cells, ar = load(arch)
        if not cells:
            continue
        orc = {c: max((t / ar[c], n, K) for (n, K), (t, a) in arms.items())
               for c, arms in cells.items() if c in ar}
        bs = {c: max(((t / ar[c], n, K) for (n, K), (t, a) in arms.items()
                      if n_active(n) <= 1))
              for c, arms in cells.items() if c in ar}
        scorers = make_scorers()
        print(f"\n===== {arch}: selection rules vs oracle "
              f"(confirm top-1 = pure ranking quality) =====")
        rows = {}
        m, w = regret(bs, orc)
        print(f"  {'best_single (C1 policy)':26s} mean {m:6.2f}%  worst {w:6.2f}%")
        rows["best_single"] = {"mean": m, "worst": w}
        for name, fn in scorers.items():
            for nc in (1, 5):
                got = rank_and_pick(cells, ar, fn, nc)
                m, w = regret(got, orc)
                print(f"  {name+' (confirm '+str(nc)+')':26s} mean {m:6.2f}%  "
                      f"worst {w:6.2f}%")
                rows[f"{name}_c{nc}"] = {"mean": m, "worst": w}
        rng = random.Random(0)
        rr = []
        for _ in range(20):
            got = {c: max((arms[k][0] / ar[c], k[0], k[1])
                          for k in rng.sample(list(arms), min(5, len(arms))))
                   for c, arms in cells.items() if c in ar}
            rr.append(regret(got, orc)[0])
        print(f"  {'random (5 draws, 20 reps)':26s} mean {sum(rr)/len(rr):6.2f}%")
        rows["random_c5"] = {"mean": sum(rr) / len(rr)}
        report[arch] = rows
    OUT.mkdir(exist_ok=True)
    (OUT / "c2_baselines.json").write_text(json.dumps(report, indent=1))
    print(f"\nwrote {OUT}/c2_baselines.json")


if __name__ == "__main__":
    main()
