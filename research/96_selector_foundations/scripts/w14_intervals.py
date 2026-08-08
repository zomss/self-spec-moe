#!/usr/bin/env python3
"""W14 — interval construction for transferred cost (the P-W14b fix).

W14/B found the elimination rule's intervals under-dispersed: the
bootstrap carried only boot-to-boot SAMPLING noise, so at b8 (0.2% boot
spread) the interval was ~0.35% wide while the systematic transfer term
was +1.4..+2.1%. Under-dispersion makes q_lo too HIGH, S_max too SMALL,
and elimination too EAGER -- a false-elimination (soundness) risk, not a
calibration nicety.

Fix: a transferred q carries TWO uncertainty sources, convolved:

    q_target = q_source x (1 + delta)

  1. sampling  -- bootstrap over BOOTS (rounds are not replicates)
  2. transfer  -- the empirical distribution of delta

An IN-REGIME measurement carries only (1). Only a REUSED measurement
pays (2). That distinction is the whole point of Round 1: measure where
you must, transfer where you can, and price the transfer honestly.

Circularity guard: delta must never be estimated from the unit being
predicted. `loo_deltas()` enforces leave-one-out; for a genuinely new
cell, delta comes from previously certified units.

Diagnostic (W14/B): delta tracks BATCH (b1 mean +0.30%, b8 mean +1.41%)
while the relative state mismatch is identical (0.56%) -- so delta is
dominated by RESIDUAL STATE MISMATCH, not content. It shrinks with
tighter state matching but cannot be assumed zero in practice.
"""
import random

DEFAULT_NBOOT = 4000


def mean(x):
    return sum(x) / len(x)


def q_point(spec_rates, ar_rates, taus):
    """q = tau / S_dec = tau / (rate_spec / rate_AR)."""
    return mean(taus) / (mean(spec_rates) / mean(ar_rates))


def q_interval(spec_rates, ar_rates, taus, deltas=None,
               nboot=DEFAULT_NBOOT, lo=2.5, hi=97.5, seed=20260808):
    """Percentile interval for q.

    deltas=None  -> in-regime: sampling uncertainty only.
    deltas=[...] -> transferred: convolved with the empirical transfer
                    distribution. MUST exclude the unit being predicted.
    """
    rng = random.Random(seed)
    vals = []
    for _ in range(nboot):
        s = [rng.choice(spec_rates) for _ in spec_rates]
        a = [rng.choice(ar_rates) for _ in ar_rates]
        t = [rng.choice(taus) for _ in taus]
        q = mean(t) / (mean(s) / mean(a))
        if deltas:
            # SYMMETRIC by construction. delta is DIRECTIONAL --
            # q_B/q_A - 1 -- so borrowing a signed delta only works if
            # the target's state offset has the same sign as the units
            # it came from. At a genuinely new cell that sign is not
            # known a priori, and applying it blind pushes the reverse
            # direction wrong by ~2*delta (measured: all four residual
            # misses in the signed version sat ABOVE their interval).
            # So use the observed MAGNITUDE with a random sign: widen
            # honestly, never shift.
            d = abs(rng.choice(deltas)) * rng.choice((-1.0, 1.0))
            q *= (1.0 + d)
        vals.append(q)
    vals.sort()
    return (q_point(spec_rates, ar_rates, taus),
            vals[int(len(vals) * lo / 100)],
            vals[int(len(vals) * hi / 100)])


def s_interval(tau_target, q_lo, q_hi):
    """S = tau/q, so the interval inverts."""
    return tau_target / q_hi, tau_target / q_lo


def s_max(q_lo, K):
    """Most favourable possible speedup, used by the elimination rule.
    Uses q_lo, which is why an under-dispersed interval is UNSAFE."""
    return (K + 1) / q_lo


def may_eliminate(q_lo, K, eps_arm=0.015):
    """Round 1 may eliminate only when even the most favourable
    acceptance cannot clear the arming rent."""
    return s_max(q_lo, K) < 1.0 + eps_arm


def loo_deltas(all_units, held_out_key, key_fn=lambda u: (u["config"],
                                                          u["batch"])):
    """Leave-one-out transfer deltas: everything EXCEPT the held-out
    unit. Enforces the circularity guard."""
    return [u["point"] for u in all_units
            if key_fn(u) != held_out_key and u.get("valid", True)]


def loo_deltas_stratified(all_units, held_out_key, stratum_fn,
                          key_fn=lambda u: (u["config"], u["batch"])):
    """Leave-one-out within a stratum (e.g. same batch), falling back to
    the global pool if the stratum is too small. delta tracks batch
    (W14/B), so stratifying is the less conservative, better-specified
    choice where data allows."""
    want = stratum_fn(held_out_key)
    strat = [u["point"] for u in all_units
             if key_fn(u) != held_out_key and stratum_fn(key_fn(u)) == want
             and u.get("valid", True)]
    return strat if len(strat) >= 2 else loo_deltas(all_units, held_out_key)
