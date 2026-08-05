#!/usr/bin/env python3
"""OURS-V2 backtest, exactly as pre-registered (README, 2026-08-05).

v2r: roofline-composed cost (aKV, aC, F fit per arch from the searched
     sample's own singles), product acceptance, UCB shortlist via the
     sound min-bound, b1 content-robustness rules.
v2g: additive cost / structural gamma {2:1.15, 3:1.33} updated online
     from the confirmations; same acceptance/shortlist/robustness.

Scored on the truth_4arch protocol: picks from each sample, judged on
the 2-sample mean S. No constant below is tuned on these oracles.
"""
import csv
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

C1 = Path("/data/smcho/self-spec-moe/research/93_c1_grid/data")
C2 = Path("/data/smcho/self-spec-moe/research/94_composition/data")
OUT = Path("/data/smcho/self-spec-moe/paper/data")

PAIRS = {
    "dense": ("oracle_dense_", "oracleR2_dense_"),
    "llama": ("oracle_llamafix_", "oracle_llamaR2_"),
    "mla":   ("oracle_mla_",   "oracleR2_mla_"),
    "moe":   ("oracle_moe_",   "oracleR2_moe_"),
}
ANCHOR = {"llama": "cells_93_llama_off.csv"}
LAYERS = {"dense": 36, "llama": 32, "mla": 27, "moe": 48}
THETA = {"none": 1.0, "hum": 0.25, "w4a16": 0.25, "w8int8": 0.5,
         "w8chan": 0.5}
GAMMA0 = {2: 1.15, 3: 1.33}      # structural prior, weight 2 (registered)
FRAGILE = ("w-512", "w-128")     # content-sensitive window components


def parse(n):
    d = {}
    for p in n.split("_"):
        k, _, v = p.partition("-")
        d[k] = v
    return d.get("q", "none"), d.get("w", "none"), d.get("s", "none")


def n_active(n):
    return sum(1 for x in parse(n) if x != "none")


def parts_of(n):
    q, w, s = parse(n)
    out = []
    for dim, val in (("q", q), ("w", w), ("s", s)):
        if val != "none":
            d = {"q": "none", "w": "none", "s": "none"}
            d[dim] = val
            out.append(f"q-{d['q']}_w-{d['w']}_s-{d['s']}")
    return out


def factors(name, ctx, layers):
    q, w, s = parse(name)
    theta = THETA[q]
    leff = min(ctx, int(w) + 16) if w != "none" else ctx
    sigma = (layers - 2) / layers if s == "b2" else 1.0
    return theta, leff, sigma


def load(prefix):
    cells = defaultdict(dict)
    for f in glob.glob(str(C2 / (prefix + "*.csv"))):
        n = Path(f).stem.split(prefix)[1]
        for r in csv.DictReader(open(f)):
            cells[(int(r["batch"]), int(r["ctx"]))][(n, int(r["K"]))] = (
                float(r["decode_toks"]), float(r["accept"] or 0))
    return cells


def R_of(t, art, tau, K):
    S = t / art
    return ((tau / S) - 1.0) / K if S > 0 and K else None


def fit_roofline(cells, ar, arch):
    """(aKV, aC, F) from the sample's own singles (incl. the base)."""
    layers = LAYERS[arch]
    rows = []
    for (b, L), arms in cells.items():
        if (b, L) not in ar:
            continue
        for (n, K), (t, a) in arms.items():
            if n_active(n) <= 1 and a > 1:
                R = R_of(t, ar[(b, L)], a, K)
                if R and 0 < R < 1.5:
                    rows.append((b, L, *factors(n, L, layers), R))
    if len(rows) < 6:
        return None

    def pred(p, b, L, th, le, sg):
        aKV, aC, F = p
        Td = max(sg * (th + aKV * b * le), sg * aC * b) + F
        Tt = max(1.0 + aKV * b * L, aC * b) + F
        return Td / Tt

    def resid(p):
        return [pred(np.abs(p), b, L, th, le, sg) - R
                for b, L, th, le, sg, R in rows]

    sol = least_squares(resid, x0=[1e-5, 0.02, 0.3], method="lm")
    return np.abs(sol.x)


def rank_v2(cells, ar, arch, variant):
    """Returns picks {cell: (S_true_in_sample, name, K)} at confirm-5."""
    layers = LAYERS[arch]
    roof = fit_roofline(cells, ar, arch) if variant == "v2r" else None
    gamma = {k: [v, 2.0] for k, v in GAMMA0.items()}   # [mean, weight]
    picks = {}
    for cell in sorted(cells):
        if cell not in ar:
            continue
        b, L = cell
        arms = cells[cell]
        R_meas, prof = {}, {}
        for (n, K), (t, a) in arms.items():
            if n_active(n) <= 1 and a > 1:
                R = R_of(t, ar[cell], a, K)
                if R and R > 0:
                    R_meas[(n, K)] = R
                prof.setdefault(n, {})[K] = a

        def R_pred(n, K):
            if n_active(n) <= 1:
                return R_meas.get((n, K))
            ps = parts_of(n)
            if variant == "v2r":
                if roof is None:
                    return None
                aKV, aC, F = roof
                th, le, sg = factors(n, L, layers)
                Td = max(sg * (th + aKV * b * le), sg * aC * b) + F
                Tt = max(1.0 + aKV * b * L, aC * b) + F
                return Td / Tt
            R0 = R_meas.get(("q-none_w-none_s-none", K))
            rs = [R_meas.get((p, K)) for p in ps]
            if R0 is None or any(r is None for r in rs):
                return None
            add = max(sum(rs) - (len(rs) - 1) * R0, 0.05)
            return add / gamma[len(ps)][0] if len(ps) in gamma else add

        def score(n, K):
            R = R_pred(n, K)
            if R is None or R <= 0:
                return None
            ps = parts_of(n)
            if not ps:                       # base or single: measured
                a = prof.get(n, {}).get(K)
                if not a or a <= 1:
                    return None
                f = (a - 1) / K
            else:                            # UCB: sound min-bound
                fs = []
                for p in ps:
                    a = prof.get(p, {}).get(K)
                    if not a or a <= 1:
                        return None
                    fs.append((a - 1) / K)
                f = min(min(fs) * 1.03, 1.0)
            s = (1 + f * K) / (K * R + 1)
            if b == 1 and any(w in n for w in FRAGILE):
                s *= 0.98                    # registered b1 list penalty
            return s

        confirmed, remaining = [], set(arms)
        for _ in range(5):
            ranked = sorted(
                ((score(n, K), n, K) for (n, K) in remaining
                 if score(n, K) is not None), reverse=True)
            if not ranked:
                break
            _, n, K = ranked[0]
            remaining.discard((n, K))
            t, a = arms[(n, K)]
            confirmed.append((t / ar[cell], n, K))
            if variant == "v2g" and n_active(n) >= 2 and a > 1:
                R_true = R_of(t, ar[cell], a, K)
                R0 = R_meas.get(("q-none_w-none_s-none", K))
                rs = [R_meas.get((p, K)) for p in parts_of(n)]
                if R_true and R0 is not None and all(rs):
                    add = max(sum(rs) - (len(rs) - 1) * R0, 0.05)
                    g = gamma.setdefault(len(rs), [1.1, 2.0])
                    g[0] = (g[0] * g[1] + add / R_true) / (g[1] + 1)
                    g[1] += 1
        if not confirmed:
            continue
        best = max(confirmed)
        if b == 1 and any(w in best[1] for w in FRAGILE):   # registered pick rule
            robust = [c for c in confirmed
                      if not any(w in c[1] for w in FRAGILE)]
            if robust and max(robust)[0] >= best[0] * 0.985:
                best = max(robust)
        picks[cell] = best
    return picks


def main():
    report = {}
    for arch in (sys.argv[1:] or list(PAIRS)):
        p1, p2 = PAIRS[arch]
        s1, s2 = load(p1), load(p2)
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
        for variant in ("v2r", "v2g"):
            r1 = reg(rank_v2({c: s1[c] for c in cells}, ar, arch, variant))
            r2 = reg(rank_v2({c: s2[c] for c in cells}, ar, arch, variant))
            rows[variant] = {"from_R1": round(r1, 3), "from_R2": round(r2, 3),
                             "mean": round((r1 + r2) / 2, 3)}
        report[arch] = rows
        print(f"=== {arch}: v2r mean {rows['v2r']['mean']:5.2f}% "
              f"({rows['v2r']['from_R1']:.2f}/{rows['v2r']['from_R2']:.2f})"
              f"   v2g mean {rows['v2g']['mean']:5.2f}% "
              f"({rows['v2g']['from_R1']:.2f}/{rows['v2g']['from_R2']:.2f})")
    OUT.mkdir(exist_ok=True)
    (OUT / "c2_ours_v2.json").write_text(json.dumps(report, indent=1))
    print(f"wrote {OUT}/c2_ours_v2.json")


if __name__ == "__main__":
    main()
