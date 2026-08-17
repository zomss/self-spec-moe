# G98-F: the knapsack layer sets transfer — and skip is still worthless alone

Record of 2026-08-16, h104 GPU 7 / NUMA 1. Runner:
`scripts/run_w98_g98f_skipsets.py`; barrier `data/g98_f/f_predictions.json`
(digest `63e1d305e6b090cf`, committed to git as `c99d069ef` **before** the
first boot); scored record `data/g98_f/f_result.json`.

D2(b) carried a non-registered reference arm showing the frozen nested skip-8
set every cost campaign booted reaches tau 5.327 where a knapsack-chosen set
of the *same count* reaches 6.423 — +20.6% acceptance at identical cost. Two
questions followed, and only measurement could answer either: does that
transfer off the regime it was chosen on, and does it make skip a lever that
stands alone?

## The circularity this design exists to avoid

The knapsack sets were **chosen from R8, content seed 4**, and confirmed on
seed 5. Scoring them on seeds 4-5 would score a selection rule on its own
selection data — the defect Phase 84 retracted a lever for. The D2 prompt
bundle contains *only* seeds 4 and 5, so the obvious path was the circular
one.

Evaluation therefore runs on the **cost lattice's seeds 2-3**
(`w98_prompt_tokens.jsonl.gz`) — disjoint from selection, and the same content
D3 itself ran on. Five of six regimes are out of domain in a second sense too,
since the sets were picked on R8 alone. That makes F1 the weak in-domain
control and **F2 the real claim**.

Six cells (`off`, `skip0`, and frozen/knapsack at counts 4 and 8), all
`target-matching` with no window so the measurement is attributable to the
skip lever, across both instrument arms — frozen and knapsack measured in the
same session so the comparison never crosses a box or a day.

## Verdict — both arms agree on every clause

| | claim | corrected | legacy |
| --- | --- | --- | --- |
| F1 | knapsack > frozen at R8 (in-domain) | **PASS** | **PASS** |
| **F2** | **knapsack > frozen in >= 4 of 6 regimes** | **PASS (5/6)** | **PASS (5/6)** |
| F3 | cost differs < 2% at the same count | FAIL | FAIL |
| F4 | knapsack skip-8 beats OFF in >= 3 of 6 | **FAIL (2/6)** | **FAIL (2/6)** |

Identical regime sets in both arms: F2 wins at R1/R4/R5/R5cot/R8 with R6 the
lone loss; F4 wins at R5cot/R8. The result is not an instrument artifact.

### F2 — the transfer holds, and it is the phase's result

Acceptance at K=4, knapsack against frozen, corrected arm:

| regime | frozen | knapsack | delta |
| --- | --- | --- | --- |
| R4 | 23.005 | 26.899 | **+16.93%** |
| R8 | 61.222 | 70.510 | **+15.17%** |
| R5 | 25.079 | 27.835 | **+10.99%** |
| R1 | 4.347 | 4.699 | **+8.09%** |
| R5cot | 31.862 | 34.230 | **+7.43%** |
| R6 | 134.526 | 126.222 | -6.17% |

A layer set chosen from one regime's leave-one-out retention improves
acceptance in five of six regimes on content it never saw, at unchanged cost.
**The knapsack sets should replace the frozen nested ones as the skip
baseline.**

The out-of-sample uplift at R8 is +15.2% against D2(b)'s in-sample +20.6% —
smaller, as a selection effect should be, and still large.

### F3 fails narrowly, and only at R4

Cost deltas: R1 +0.87%, R4 **+2.17%**, R5 -1.02%, R5cot -0.90%, R6 -0.74%,
R8 -0.00% (legacy: +0.11 / **+2.28** / -1.39 / -1.11 / -0.90 / +0.04). Five of
six are inside the registered 2%; R4 clears it by a fifth of a point in both
arms, so it is consistent rather than noise. The sets remove the same NUMBER
of layers, so this is not a FLOP difference — the knapsack set is contiguous
(6,7,8,9,10) where the frozen one is spread, and a contiguous run plausibly
changes access patterns. Small, real, unexplained.

### F4 fails — and it refutes this record's own estimate

`results_lever_mechanics.md` estimated that knapsack sets would take skip-8
from beating no-speculation in 1 of 6 regimes to **4 of 6**, and withdrew the
phase's "window and skip are worthless alone" on that basis. Measured: **2 of
6**, where the frozen set manages 3.

| regime | frozen x OFF | knapsack x OFF |
| --- | --- | --- |
| R5cot | 1.050 | **1.135** |
| R8 | 1.024 | **1.174** |
| R6 | 1.053 | 0.995 |
| R5 | 0.843 | 0.942 |
| R4 | 0.724 | 0.786 |
| R1 | 0.795 | 0.763 (host-contaminated) |

Better acceptance is not enough: skip-8 still costs ~79% of a base draft that
itself loses to OFF at 0.891x. What the knapsack sets buy is **bigger wins
where skip already won** (R5cot 1.050 -> 1.135, R8 1.024 -> 1.174), not new
ones. The withdrawal is **reversed**; "skip is worthless alone" stands.

The error is worth naming, because it is the second of its kind today: a
single-regime factor extrapolated uniformly and pushed through the FITTED
cost map instead of being measured. The Marlin bound failed the same way. Both
times the fitted route produced a plausible number that measurement
contradicted — which is, in fairness, this phase's own thesis about
acceptance applied to itself.

## What still changes

The surviving consequence from that section: the single-lever skip baseline
was handicapped by 7-17% of acceptance in five of six regimes, so the value
decomposition's "**composing is worth +12.5% over the best single lever**" is
measured against a depressed single and should be recomputed on the knapsack
baseline. The correction will be small — quantization, not skip, is the best
single lever — but it should be made.

## Limitations

* **One boot per cell.** Acceptance is boot-deterministic (verified 17/17
  requests, 0/210 steps across boots), and the frozen arm's tau is
  bit-identical between instrument arms here, which corroborates it. Rates
  are not deterministic and carry the usual boot variance.
* **R1 rates are host-contaminated.** Step time 49.4 ms against ~30.1 implied
  by chain+verify, from co-resident CPU jobs on the same box. The
  `draft_chain` gate PASSES on every cell — monotonic in batch, normal spread
  — because the contamination is in inter-step host time, which that gate
  does not observe. F4 is therefore 2 of 5 clean regimes.
* **The runner applies `hostload.measurement_verdict` post-hoc**, not inline
  at boot as the scored campaigns do. A re-run should wire it in so a
  contaminated boot is rejected rather than scored.
* One model, one box, K=4, seeds 2-3.
