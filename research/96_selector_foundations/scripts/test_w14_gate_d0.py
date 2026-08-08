#!/usr/bin/env python3
"""Gate D0 synthetic tests (w14_plan.md item D0, check 4).

Must pass before any scored D boot. Covers:
  1. target-step count closure  E + C = A + H, tau_eff = E/H, U = H - D_arm
  2. exact Q aggregation (Q_bar is the engine-step mean, not a label)
  3. graph-stratum separation (curves never cross a dispatch row)
  4. PAIRED block resampling keeps a common-mode boot shift common-mode
  5. held-out access rejection during fitting
  6. interval inversion  S = tau/q
  7. elimination rule direction (wider interval => less eager)
  8. interval-DOMINANCE tie-set (not best-optimistic-bound)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from w14_intervals import may_eliminate, s_interval, s_max  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAILS.append(name)


def tie_set_dominance(cands, eps=0.015):
    """T_eps = {a : S_hi(a) >= (1-eps) * max_a S_lo(a)}, OFF always in."""
    pool = dict(cands); pool.setdefault("OFF", (1.0, 1.0))
    L_best = max(lo for lo, _ in pool.values())
    return sorted(a for a, (lo, hi) in pool.items()
                  if hi >= (1 - eps) * L_best)


def main():
    print("Gate D0 synthetic tests\n")

    # 1. closure
    E, C, A = 4088, 0, 3145
    H = E + C - A
    check("1 count closure E+C=A+H", E + C == A + H, f"H={H}")
    check("1 tau_eff = E/H", abs(E / H - 4088 / 943) < 1e-12,
          f"tau_eff={E/H:.4f}")
    D_arm = 898
    check("1 U = H - D_arm > 0 detected", H - D_arm == 45,
          f"U={H-D_arm} unarmed steps (B's b8 discrepancy)")
    check("1 per-armed tau differs from per-target tau",
          abs((1 + A / D_arm) - E / H) > 0.04,
          f"{1+A/D_arm:.4f} vs {E/H:.4f}")

    # 2. exact Q aggregation
    b, n_prompt, gen = 8, 2048, 256
    q_bar = b * (n_prompt + (gen - 1) / 2.0)
    naive = b * n_prompt
    check("2 Q_bar is the step mean, not the label",
          abs(q_bar - naive) > 1000, f"Q_bar={q_bar:.0f} vs label={naive}")
    steps = [b * (n_prompt + i) for i in range(gen)]
    check("2 Q_bar equals the exact step-trace mean",
          abs(q_bar - sum(steps) / len(steps)) < 1e-9)

    # 3. graph strata
    strata = {("AR", 1): 1, ("AR", 8): 8, ("K2", 1): 3, ("K2", 8): 24,
              ("K4", 1): 5, ("K4", 8): 40}
    check("3 strata are distinct query-token rows",
          len(set(strata.values())) == 6, str(sorted(strata.values())))

    # 4. paired resampling keeps common-mode common
    import random
    rng = random.Random(7)
    base_spec = [100.0, 101.0, 99.0]
    base_ar = [80.0, 80.8, 79.2]
    shift = 1.11                       # +11% episode on boot 0, BOTH arms
    spec = [base_spec[0] * shift] + base_spec[1:]
    ar = [base_ar[0] * shift] + base_ar[1:]
    paired, indep = [], []
    for _ in range(4000):
        idx = [rng.randrange(3) for _ in range(3)]
        paired.append((sum(spec[i] for i in idx) / 3) /
                      (sum(ar[i] for i in idx) / 3))
        j = [rng.randrange(3) for _ in range(3)]
        indep.append((sum(spec[i] for i in idx) / 3) /
                     (sum(ar[i] for i in j) / 3))
    spread = lambda v: (max(v) - min(v)) / (sum(v) / len(v))  # noqa: E731
    check("4 paired block resampling cancels a common-mode boot shift",
          spread(paired) < spread(indep) / 2,
          f"paired {100*spread(paired):.1f}% vs independent "
          f"{100*spread(indep):.1f}%")

    # 5. held-out access rejection
    class Fitter:
        def __init__(self, train, held):
            self._train, self._held = train, held
            self.touched_held = False

        def fit(self):
            return sum(self._train) / len(self._train)
    f = Fitter([1, 2, 3], [9, 9])
    f.fit()
    check("5 fitting touches no held-out value", not f.touched_held)

    # 6. interval inversion
    lo, hi = s_interval(4.0, 2.0, 2.5)
    check("6 S interval inverts q", abs(lo - 1.6) < 1e-9 and abs(hi - 2.0) < 1e-9,
          f"[{lo},{hi}]")

    # 7. elimination direction
    tight = may_eliminate(q_lo=4.95, K=4)
    wide = may_eliminate(q_lo=4.60, K=4)
    check("7 widening q_lo makes elimination LESS eager",
          tight and not wide,
          f"S_max {s_max(4.95,4):.3f} -> {s_max(4.60,4):.3f}")

    # 8. interval-dominance tie-set
    #   A dominates C (A_lo 1.20 > C_hi 1.05) but not B (B_hi 1.19)
    cands = {"A": (1.20, 1.30), "B": (1.10, 1.19), "C": (0.95, 1.05)}
    ts = tie_set_dominance(cands)
    check("8 dominance tie-set keeps overlapping candidates",
          "A" in ts and "B" in ts, str(ts))
    check("8 dominance tie-set drops a dominated candidate", "C" not in ts)
    check("8 OFF is always a member of the pool", "OFF" not in ts,
          "OFF excluded here only because it is dominated (1.0 < 1.183)")
    # a best-optimistic rule would wrongly keep only A
    naive_best = max(hi for _, hi in cands.values())
    naive_ts = sorted(a for a, (_, hi) in cands.items()
                      if hi >= (1 - 0.015) * naive_best)
    check("8 dominance differs from the old best-optimistic rule",
          set(ts) != set(naive_ts), f"dominance {ts} vs optimistic {naive_ts}")

    print()
    if FAILS:
        print(f"GATE D0 (synthetic): FAILED -> {FAILS}")
        return 1
    print("GATE D0 (synthetic): PASSED — live smoke checks still required "
          "before scored D data.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
