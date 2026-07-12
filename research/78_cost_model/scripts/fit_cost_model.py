#!/usr/bin/env python3
"""Fit the Phase-78 term-decomposed cost model to 76's measured TPOTs.

  T(group, arm, b, ctx) = F(n) + mem_or_roofline + C_comm(n) + kappa(arm)
    F(n)       = f0 + f1*n                (n = per-rank batch; launch floor)
    mem        = [W_read(arm, n) + KV_read(arm, n, ctx)] / BW_eff
    roofline   = max(mem, 2*P_active*n / peak)          (--form roofline)
    C_comm(n)  = c0 + c1*n                (MoE/EP only; zeroed by localroute
                                           and skip-a2a arms)
    kappa(arm) = per-quant-kernel overhead (Marlin dequant tax etc.)

Levers enter ONLY as term edits; architectures ONLY as config constants
(weights, kv bytes/token, expert geometry) -- see CONST/ARMS tables.
MoE expert weight-read uses the distinct-activated-expert expectation:
residents * (1 - (1 - k/E)^tokens) * expert_bytes  (uniform-routing
simplification; P21 measured skew makes this an overestimate at small n).

V0 gate: median |rel err| <= 5% per group AND measured crossovers reproduced.
V1 gate (--holdout win): window rows excluded from fit, predicted from KV-byte
accounting alone; report held-out error (<= 10%).

Writes data/fit.json. Fit points: absolute TPOTs incl. reconstructed bf16
anchors (T_bf16 = tpot/ratio, overpool rows excluded from reconstruction but
arm rows kept -- the arm's own TPOT is valid even when the bf16 denom thrashed).
"""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

PHASE = Path(__file__).resolve().parent.parent
P76 = PHASE.parent / "76_lever_latency_sweep"

GB = 1e9
CONST = {
    "dense": dict(
        W=15.24e9, kv_tok=57344.0, P_active=7.62e9, dp=1,
        ctx_of={2: 2048, 16: 16384, 32: 32512}),
    "moe": dict(
        W_dense=3.07e9, n_res=32, E=128, topk=8, exp_bytes=9.44e6,
        kv_tok=98304.0, P_active=3.3e9, dp=4,
        ctx_of={2: 2048, 16: 16384, 32: 32768}),
}
GEN_MEAN = 64          # measured TPOT averages over 128 generated tokens
WIN_KV = 512 + 16      # window + sinks

# arm -> term edits: w=weight-read scale, kvs=KV scale, win=KV cap,
# ls=layer multiplier, comm=EP comm on, kappa=fitted-overhead key
ARMS = {
    ("dense", "bf16"): dict(),
    ("dense", "d_w4marlin"): dict(w=0.28, kappa="marlin"),
    ("dense", "d_w4machete"): dict(w=0.28, kappa="machete"),
    ("dense", "d_fp8w8a8"): dict(w=0.5, kappa="fp8d"),
    ("dense", "d_kvq"): dict(kvs=0.5),
    ("dense", "d_win"): dict(win=WIN_KV),
    ("dense", "d_skip25"): dict(ls=0.75),
    ("dense", "d_skip50"): dict(ls=0.5),
    ("moe", "bf16"): dict(comm=True),
    ("moe", "m_fp8marlin"): dict(w=0.5, kappa="fp8m", comm=True),
    ("moe", "m_fp8block"): dict(w=0.5, kappa="fib", comm=True),
    ("moe", "m_kvq"): dict(kvs=0.5, comm=True),
    ("moe", "m_win"): dict(win=WIN_KV, comm=True),
    ("moe", "m_skip50"): dict(ls=0.5, comm=True),
    ("moe", "m_localroute"): dict(comm=False, local=True),
    ("moe", "m_skipa2a"): dict(comm=False, kappa="a2atile"),
}
KAPPAS = {"dense": ["marlin", "machete", "fp8d"], "moe": ["fp8m", "fib", "a2atile"]}


def load_points():
    pts = defaultdict(list)   # group -> [(arm, b_global, ctx_k, tpot)]
    recon = defaultdict(list)  # (group, cell) -> [T_bf16 reconstructed]
    for r in csv.DictReader((P76 / "data/e1/summary.csv").open()):
        g, a = r["group"], r["arm"]
        if g not in CONST or (g, a) not in ARMS and a not in ("d_bf16dummy", "m_bf16dummy"):
            if (g, a) not in ARMS:
                continue
        cell = (int(r["batch"]), int(r["ctx"]))
        t, ratio, op = float(r["tpot_ms"]), float(r["ratio"]), int(r["overpool"])
        if (g, a) in ARMS:
            if not op:
                pts[g].append((a, *cell, t))
            if r["denom"].endswith("_bf16") and not op:
                recon[(g, cell)].append(t / ratio)
    for (g, cell), vals in recon.items():
        pts[g].append(("bf16", *cell, float(np.median(vals))))
    return pts


def w_read(g, spec, n_rank, b_global):
    c = CONST[g]
    ws = spec.get("w", 1.0)
    if g == "dense":
        return c["W"] * ws
    if spec.get("local"):
        distinct = c["n_res"] * (1 - (1 - c["topk"] / c["n_res"]) ** n_rank)
    else:
        distinct = c["n_res"] * (1 - (1 - c["topk"] / c["E"]) ** b_global)
    return (c["W_dense"] + distinct * c["exp_bytes"]) * ws


def kv_read(g, spec, n_rank, ctx):
    c = CONST[g]
    L = ctx + GEN_MEAN
    if "win" in spec:
        L = min(L, spec["win"] + GEN_MEAN)
    return n_rank * L * c["kv_tok"] * spec.get("kvs", 1.0)


def predict(g, params, arm, b_global, ctx_k, form):
    c = CONST[g]
    spec = ARMS[(g, arm)]
    n = b_global / c["dp"]
    ctx = c["ctx_of"][ctx_k]
    kap = KAPPAS[g]
    f0, f1, bw, h = params[0], params[1], params[2], params[3]
    i = 4
    c0 = c1 = 0.0
    if g == "moe":
        c0, c1 = params[4], params[5]
        i = 6
    kappas = dict(zip(kap, params[i:i + len(kap)]))
    peak = params[i + len(kap)] if form == "roofline" else None

    ls = spec.get("ls", 1.0)
    mem_ms = (w_read(g, spec, n, b_global) + kv_read(g, spec, n, ctx)) / bw * 1e3 / GB
    # KV-MANAGEMENT term: scales with ALLOCATED context (block tables /
    # attention metadata), NOT with bytes read -- window/kvq don't cut it
    # (first fit under-predicted exactly the big-allocated-KV cells by 33-49%)
    t = (f0 + f1 * n + h * n * (ctx + GEN_MEAN) / 1e6) * ls
    if form == "roofline":
        comp_ms = 2 * c["P_active"] * n / (peak * 1e12) * 1e3
        t += max(mem_ms, comp_ms) * ls
    else:
        t += mem_ms * ls
    if spec.get("comm"):
        t += c0 + c1 * n
    t += kappas.get(spec.get("kappa", ""), 0.0)
    return t


def fit_group(g, points, form, holdout=None):
    fit_pts = [p for p in points if not (holdout and p[0] == holdout)]
    kap = KAPPAS[g]
    x0 = np.array([3.0, 0.2, 2000.0, 10.0] + ([1.0, 0.2] if g == "moe" else [])
                  + [0.3] * len(kap) + ([300.0] if form == "roofline" else []))

    def resid(x):
        return [(predict(g, x, a, b, c, form) - t) / t for (a, b, c, t) in fit_pts]

    sol = least_squares(resid, x0, bounds=(0, np.inf), max_nfev=20000)
    return sol.x


def report(g, params, points, form, holdout=None):
    errs, rows = [], []
    for (a, b, c, t) in points:
        p = predict(g, params, a, b, c, form)
        e = (p - t) / t
        errs.append((abs(e), a, b, c, t, p))
        rows.append(dict(arm=a, batch=b, ctx=c, tpot=t, pred=round(p, 2),
                         err_pct=round(100 * e, 1)))
    med = float(np.median([e[0] for e in errs]))
    if holdout:
        ho = [e for e in errs if e[1] == holdout]
        ho_med = float(np.median([e[0] for e in ho])) if ho else float("nan")
        print(f"[{g}/{form}] V1 held-out '{holdout}': median |err| {100*ho_med:.1f}% "
              f"({'PASS' if ho_med <= 0.10 else 'FAIL'} @10%)")
        return rows, med
    print(f"[{g}/{form}] V0 in-grid median |err| {100*med:.1f}% "
          f"({'PASS' if med <= 0.05 else 'FAIL'} @5%)  worst:")
    for e in sorted(errs, reverse=True)[:3]:
        print(f"    {e[1]} b{e[2]}/{e[3]}k meas {e[4]:.2f} pred {e[5]:.2f} ({100*e[0]:.0f}%)")
    return rows, med


def crossovers(g, params, points, form):
    """V0 gate second half: predicted winner ordering matches measured."""
    meas = {(a, b, c): t for (a, b, c, t) in points}
    ok, tot = 0, 0
    pairs = ([("d_win", "d_w4marlin")] if g == "dense" else [("m_win", "bf16")])
    for (wa, qa) in pairs:
        for (a, b, c, t) in points:
            if a != wa or (qa, b, c) not in meas:
                continue
            tot += 1
            m_order = t < meas[(qa, b, c)]
            p_order = (predict(g, params, wa, b, c, form)
                       < predict(g, params, qa, b, c, form))
            ok += (m_order == p_order)
    print(f"[{g}/{form}] crossover orderings reproduced: {ok}/{tot}")
    return ok, tot


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--form", choices=["additive", "roofline", "both"], default="both")
    ap.add_argument("--holdout", default=None,
                    help="arm to exclude from fit and predict (V1), e.g. d_win")
    a = ap.parse_args()
    pts = load_points()
    forms = ["additive", "roofline"] if a.form == "both" else [a.form]
    out = {}
    for form in forms:
        for g in ("dense", "moe"):
            params = fit_group(g, pts[g], form, holdout=a.holdout)
            rows, med = report(g, params, pts[g], form, holdout=a.holdout)
            ok, tot = crossovers(g, params, pts[g], form)
            kap = KAPPAS[g]
            names = (["f0", "f1", "BW_GBs", "h_ms_per_Mtok"]
                     + (["c0", "c1"] if g == "moe" else [])
                     + [f"kappa_{k}" for k in kap]
                     + (["peak_TFLOPs"] if form == "roofline" else []))
            out[f"{g}/{form}"] = dict(
                params={n: round(float(v), 4) for n, v in zip(names, params)},
                median_err=round(med, 4), crossovers=f"{ok}/{tot}",
                holdout=a.holdout, residuals=rows)
            print(f"    params: " + " ".join(
                f"{n}={v:.3g}" for n, v in zip(names, params)))
    (PHASE / "data").mkdir(exist_ok=True)
    f = PHASE / ("data/fit.json" if not a.holdout else f"data/fit_holdout_{a.holdout}.json")
    f.write_text(json.dumps(out, indent=1))
    print(f"wrote {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
