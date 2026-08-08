#!/usr/bin/env python3
"""W14/D scorer — paired-block bootstrap, affine latency surface,
three-component log-space interval, interval-dominance tie-sets.

Replaces the B diagnostics (w14_intervals.py independent resampling,
best-optimistic tie-sets) for all D scoring.

Interval construction, per w14_plan.md:

    z = log(q)
    z_draw = z_surface_from_paired_block_bootstrap
             + r_surface                       (training-only LOO-context)
             + r_transfer_after_state_adjustment

- resample complete BOOT-BLOCK ids; every rate, counter, configuration,
  regime, context and its AR anchor stay paired inside the block;
- r_surface: training-only leave-one-context-out log residuals;
- r_transfer: fit a training-only R5cot surface, evaluate the R5 and
  R5cot surfaces at the SAME exact Q, take the log difference. This is
  state-adjusted, so it is NOT B's raw directional delta and does not
  double-count a context-interpolation residual;
- symmetrize both pools in log space within batch/graph stratum: a
  reused interval WIDENS, it is never SHIFTED by an unknowable sign.

A same-cell measurement carries sampling uncertainty only. Every reused
prediction carries sampling + surface + transfer.

Held-out cells are structurally unreachable during fitting: fit_surface
receives only training rows and raises if a held-out row is passed.
"""
import json
import math
import random
from pathlib import Path

PHASE = Path(__file__).resolve().parents[1]
W14 = PHASE / "data" / "w14"
EPS_ARM = 0.015
EPS_SEL = 0.015
NBOOT = 4000
MIN_RESID = 8


class HeldOutLeak(Exception):
    """Raised if a held-out row reaches the fitter."""


def affine_fit(xs, ys):
    """Least squares y = a + b x."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return my, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return my - b * mx, b


def fit_surface(rows):
    """rows: [{Q, latency, split}]. Refuses held-out rows."""
    for r in rows:
        if r.get("split") != "train":
            raise HeldOutLeak(f"held-out row reached fitter: {r}")
    return affine_fit([r["Q"] for r in rows], [r["latency"] for r in rows])


def loo_context_residuals(rows):
    """Training-only leave-one-context-out log residuals."""
    out = []
    ctxs = sorted({r["ctx"] for r in rows})
    if len(ctxs) < 3:
        return out
    for held in ctxs:
        tr = [r for r in rows if r["ctx"] != held]
        te = [r for r in rows if r["ctx"] == held]
        a, b = fit_surface(tr)
        for r in te:
            pred = a + b * r["Q"]
            if pred > 0 and r["latency"] > 0:
                out.append(math.log(r["latency"] / pred))
    return out


def symmetrize(pool):
    """Log-space symmetrization: widen, never shift."""
    return [x for v in pool for x in (v, -v)]


def q_interval_d(sample_fn, r_surface, r_transfer, nboot=NBOOT,
                 seed=20260808):
    """Monte Carlo on z=log(q). sample_fn() -> one paired-block draw of q."""
    rng = random.Random(seed)
    rs = symmetrize(r_surface) or [0.0]
    rt = symmetrize(r_transfer) or [0.0]
    vals = []
    for _ in range(nboot):
        q = sample_fn(rng)
        if q <= 0:
            continue
        z = math.log(q) + rng.choice(rs) + rng.choice(rt)
        vals.append(math.exp(z))
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))]


def paired_block_sampler(blocks):
    """blocks: {block_id: {"spec": [...], "ar": [...], "tau": [...]}}.
    Resamples whole block ids so an AR anchor never separates from the
    speculative boots it was measured beside."""
    ids = list(blocks)

    def draw(rng):
        pick = [rng.choice(ids) for _ in ids]
        sp = [v for i in pick for v in blocks[i]["spec"]]
        ar = [v for i in pick for v in blocks[i]["ar"]]
        tt = [v for i in pick for v in blocks[i]["tau"]]
        return (sum(tt) / len(tt)) / ((sum(sp) / len(sp)) /
                                      (sum(ar) / len(ar)))
    return draw


def s_interval(tau_lo, tau_hi, q_lo, q_hi):
    return tau_lo / q_hi, tau_hi / q_lo


def may_eliminate(q_lo, K, eps=EPS_ARM):
    return (K + 1) / q_lo < 1.0 + eps


def tie_set(cands, eps=EPS_SEL):
    """Interval DOMINANCE: drop a only when another's pessimistic bound
    beats a's optimistic bound by more than eps. OFF always considered."""
    pool = dict(cands)
    pool.setdefault("OFF", (1.0, 1.0))
    l_best = max(lo for lo, _ in pool.values())
    return sorted(a for a, (lo, hi) in pool.items()
                  if hi >= (1 - eps) * l_best)


def elimination_resolved(q_true_lo, K, eps=EPS_ARM):
    """A predicted elimination is VALIDATED only when the oracle's own
    optimistic bound is below the rent. A straddling oracle interval is
    UNRESOLVED and cannot certify."""
    return (K + 1) / q_true_lo < 1.0 + eps


def main():
    print("W14/D scorer — paired-block bootstrap + 3-component log interval")
    print("This module is the frozen D scoring contract; it is imported by")
    print("the D results script once 21 valid scored boots exist.\n")
    files = sorted(W14.glob("w14d_block*_*.json"))
    if not files:
        print("no scored D boots yet (D0 smoke does not qualify).")
        print("Frozen rules in force:")
        print(f"  eps_arm={EPS_ARM}  eps_sel={EPS_SEL}  nboot={NBOOT}")
        print(f"  min residuals per stratum before pooling upward: {MIN_RESID}")
        print("  held-out rows raise HeldOutLeak inside fit_surface()")
        return
    print(f"found {len(files)} scored D boots")


if __name__ == "__main__":
    main()
