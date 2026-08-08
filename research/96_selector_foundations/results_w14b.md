# W14/B results — cost transfers at a matched state; intervals do not

Pre-registration `w14_plan.md` item B (d57f81222). Scorer
`score_w14b.py`, committed before the data (38eb651e4). Artifacts
`data/w14/w14b_*.json`, scored `data/w14/w14b_scored.json`.

Boots: **12/12 complete** (AR 3, w512 3, w2048 3, woff 3). w512's third
boot needed six attempts across two retry rounds against the draft
graph-capture wedge. All units satisfy the matched-work rule: total-KV
mismatch 0.56%, inside the 1% validity bound.

## P-W14a — CERTIFIED 6/6 (needed ≥5/6)

95% boot-level CI on `(q_R5cot / q_R5) − 1`:

| unit | point | 95% CI | verdict |
|---|---|---|---|
| w512 b1 | −0.1% | [−2.6, +2.4] | PASS |
| w512 b8 | +1.7% | [+1.5, +2.0] | PASS |
| w2048 b1 | +0.8% | [−1.7, +2.9] | PASS |
| w2048 b8 | +1.7% | [+1.4, +2.0] | PASS |
| woff b1 | +0.2% | [−0.7, +1.1] | PASS |
| woff b8 | +0.8% | [+0.5, +1.2] | PASS |

**Transfer mask: all six units certified.**

The e2e-currency baseline was 8.4% mean / 17.7% max (W13). Decode
currency reduces it to 0.2–1.9%, confirming that W13's apparent
"transfer failure" was prefill dilution in the meter, not a property of
the system.

**The dichotomy, measured in one controlled experiment.** At matched
execution state, across the same six units:

- acceptance differs by up to **16%** between regimes (τ = 4.248 vs
  4.919 at w512 b1);
- cost transfers within **2%** (q = 3.335 vs 3.324, i.e. −0.1%).

Cost is content-free at a matched state; acceptance is workload-local.
That is the premise the two-round split rests on, previously inferred
across phases and now measured directly.

## P-W14b — SPLIT, and the failure is informative

**Selector evaluations: 4/4 PASS, zero regret.** In every evaluation
the predicted ε-optimal tie-set contains the measured best
configuration and simple regret is 0.00%.

**Directional containment: 6/12 — FAILS the registered ≥11/12 bar.**
The pattern is perfectly stratified: **all 6 passes are at b1, all 6
failures at b8.** Every failure is a NEAR miss (0.5–1.8% outside the
interval), and the point predictions are good throughout.

The cause is **under-dispersed intervals, not an inaccurate model.**
The bootstrap CI carries only boot-to-boot sampling noise. At b8 the
measurement is extraordinarily reproducible — boot spread 0.2% — so the
interval is ~0.35% wide (e.g. [1.692, 1.698]) while the systematic
transfer term is the +1.4…+2.1% that P-W14a itself measures. The
interval is narrower than the bias it must cover, so containment fails
even though the prediction is within 1.8%.

## The consequence: this is a SOUNDNESS bug, not a calibration nicety

Round 1 eliminates when `S_max = (K+1)/q_lo < 1 + ε_arm`. An
under-dispersed interval makes **q_lo too high**, hence `S_max` too
small, hence elimination too eager. With ε_arm = 1.5% and a transfer
term of 1–2%, an interval that omits it can produce a **FALSE
ELIMINATION** — precisely the soundness property the design claims and
that P-W14d is meant to test.

**Required fix**: Round 1's `q` interval must be inflated by a
transfer-uncertainty term before the elimination rule is applied, not
only by sampling error. The magnitude is now measured (≤2.1% across all
six certified units), but it must be validated on held-out cells in D/E
rather than taken from the same data that measured it — using P-W14a's
own numbers to widen P-W14b's intervals would be circular, so the
registered 6/12 result stands as a FAIL and is not rescored.

## Reading

- The **transfer claim** is supported: cost is reusable across content
  at a matched execution state, on all six dense units tested.
- The **decision machinery** works: 4/4 selector evaluations, zero
  regret.
- The **uncertainty quantification** is not yet fit for the elimination
  rule, and that is a blocking defect for the soundness claim, found
  before it could cause a wrong elimination.

## Caveats

- All results above are the FINAL 12-boot scoring. Adding w512's third
  boot moved its b1 point estimate −0.3% → −0.1% and WIDENED its CI
  ([−2.4,+1.8] → [−2.6,+2.4]) — the extra boot revealed more spread
  than two boots had shown, which is the argument for n=3 made
  concrete. No verdict changed; one selector evaluation's measured best
  moved within an existing tie-set (w512 → w2048, regret still 0.00%).
- `woff` remains a distinct realization, not a clean window
  intervention (plan §"Scope corrections"); its agreement is evidence
  about that realization, not about window size alone.
- Dense only. No architecture-wide claim follows.
