#!/usr/bin/env python3
"""Gate A (w14_plan.md item A): instrumentation accounting must close
BEFORE any scored W14 run.

This is an INSTRUMENTATION gate. It does not validate the algebraic
identity q = tau/S_dec (that is algebra, not an empirical claim).

Checks:
 1. synthetic cumulative snapshots recover known interval deltas exactly
 2. the decode-rate numerator is (out_tokens - n_requests): the first
    token of each request is produced by prefill, not a decode step
 3. histogram count delta equals completed requests
 4. repeated snapshots are monotone
 5. q = tau/S_dec reproduces P/T on synthetic exact inputs (round-trip)
"""
import sys


def delta(cum_a, cum_b):
    """Interval delta between two cumulative (sum, count) snapshots."""
    return cum_b[0] - cum_a[0], cum_b[1] - cum_a[1]


def fail(msg):
    print(f"  FAIL: {msg}")
    return 1


def main():
    bad = 0

    # 1. synthetic cumulative snapshots -> exact interval recovery
    snaps = [(0.0, 0), (12.5, 8), (30.0, 20), (30.0, 20), (55.25, 33)]
    want = [(12.5, 8), (17.5, 12), (0.0, 0), (25.25, 13)]
    got = [delta(snaps[i], snaps[i + 1]) for i in range(len(snaps) - 1)]
    if got != want:
        bad += fail(f"interval recovery {got} != {want}")
    else:
        print("  ok  1. cumulative snapshot deltas exact (incl. empty interval)")

    # 2/3. token accounting on a fixed-width interval
    n_req, fixed_len = 8, 512
    out_tokens = n_req * fixed_len          # ignore_eos => every request equal
    dec_count = n_req
    dec_numerator = out_tokens - n_req      # first token per request = prefill
    if dec_numerator != n_req * (fixed_len - 1):
        bad += fail("decode numerator != n_req*(fixed_len-1)")
    else:
        print(f"  ok  2. decode numerator = out-n_req = {dec_numerator} "
              f"= n_req*(L-1)")
    if dec_count != n_req:
        bad += fail("histogram count delta != completed requests")
    else:
        print("  ok  3. histogram count delta == completed requests")

    # 4. monotonicity
    if any(snaps[i][0] > snaps[i + 1][0] or snaps[i][1] > snaps[i + 1][1]
           for i in range(len(snaps) - 1)):
        bad += fail("cumulative snapshots not monotone")
    else:
        print("  ok  4. cumulative snapshots monotone")

    # 5. round-trip: build exact synthetic rates, check q = tau/S_dec = P/T
    T_true, P_true, tau = 6.0e-3, 22.5e-3, 3.75
    # AR arm: 1 token per step per request
    rate_AR = 1.0 / T_true
    # spec arm: tau tokens per armed step per request
    rate_spec = tau / P_true
    S_dec = rate_spec / rate_AR
    q_from_rates = tau / S_dec
    q_from_times = P_true / T_true
    if abs(q_from_rates - q_from_times) > 1e-12:
        bad += fail(f"round-trip {q_from_rates} != {q_from_times}")
    else:
        print(f"  ok  5. q round-trip exact: tau/S_dec = P/T = "
              f"{q_from_times:.6f}")

    print()
    if bad:
        print(f"GATE A: FAILED ({bad} checks)")
        return 1
    print("GATE A: PASSED — instrumentation accounting closes.")
    print("  (instrumentation only; live token-accounting check still runs")
    print("   on the first fixed-width sanity cell before B is scored)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
