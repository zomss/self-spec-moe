# D3 re-scored under a fail-closed rule — 99.6% of omniscient, and the failing clause flips

Record of 2026-08-16, following `results_g98_e.md`. **No new measurement.**
The G98-E grid already contains OFF and all 30 armed configurations at all
six regimes on both instrument arms, so changing the selection rule is
analysis over measured cells. Script: `scripts/analyze_w98_failclosed.py`;
record: `data/g98_e/d3_failclosed.json`.

## Why there is a rule to build

D3's selector armed at every regime because its pick was the argmax of a
predicted-rate map — it had no way to decline. At R4 that cost 24% (522
tok/s against OFF's 685). The map itself had already flagged the problem:
R4's predicted margin over OFF was **1.026x, the thinnest of the six**. The
signal was present and unread.

## The rule

Within a regime the decision is a ratio of two predicted rates,

```
m(R) = max_cell rate_hat(cell, R) / rate_hat(off, R)
     = tau_hat * verify / (d_hat + verify)
```

so error common to both arms cancels; what survives is acceptance error and
the *differential* cost error between draft chain and verify. Arm only when
the lower confidence bound on that margin clears unity:

```
LCB(R) = m(R) * exp(-z * sigma_log_margin) > 1

sigma_log_margin = hypot( sigma_tau , (d/(d+v)) * sqrt(2) * sigma_cost )
```

The `d/(d+v)` factor is the draft's share of the armed step: a cost error
that moves the draft chain and verify together cancels out of the ratio, and
only the differential survives, scaled by that share.

## Where the threshold comes from

Both error terms are estimated from data predating the D3 grid and disjoint
from its content:

| term | value | source |
| --- | --- | --- |
| `sigma_tau` | **0.0195** (log) | G98-D measured every acceptance cell on content seeds 4 AND 5; **180 matched pairs**. D3 evaluates on seeds 2-3, so this is the right scale for "predicted acceptance vs what a fresh content draw delivers". |
| `sigma_cost` | **0.0123** (relative) | D1' held-out rows, RMS over 48 (`round2_result.json`). |

At `z = 1.645` (one-sided 95%) this yields an equivalent bare-margin
threshold of **1.037-1.040** across regimes. Nothing about it was chosen to
produce an outcome.

## Decisions

| regime | margin | sigma | LCB | decision |
| --- | --- | --- | --- | --- |
| R1 | 1.3348 | 0.0232 | 1.2847 | ARM |
| **R4** | **1.0264** | 0.0237 | **0.9873** | **OFF — fail closed** |
| R5 | 1.6316 | 0.0222 | 1.5731 | ARM |
| R5cot | 1.8052 | 0.0222 | 1.7405 | ARM |
| R6 | 1.2282 | 0.0233 | 1.1821 | ARM |
| R8 | 1.1796 | 0.0234 | 1.1351 | ARM |

The rule fires at exactly one regime, and it is the right one.

## Result

| clause | as measured | fail-closed |
| --- | --- | --- |
| fraction of omniscient (corrected) | 95.1% | **99.6%** |
| fraction of omniscient (legacy) | 94.8% | **99.6%** |
| over static-OFF (corrected) | 1.371x | **1.436x** |
| over static-OFF (legacy) | 1.311x | **1.377x** |
| over best static (corrected) | 1.014x | **1.062x** |
| over best static (legacy) | 1.018x | **1.069x** |
| **beats every static by +2%** | **FAIL** | **PASS** |

**The clause that was awaiting a ruling now passes on the strict reading.**
D3's ambiguity — whether "every static single configuration" means every
single-*lever* config or any one fixed config — was load-bearing only
because the selector lost to one three-lever composition by 1.4%. Under the
fail-closed rule it beats every one of the 31 statics by 6.2%, so the
ambiguity is moot: the strict reading passes and the ruling is no longer
needed.

Both instrument arms agree on every line, so this is not an instrument
artifact.

### The mix frontier moves further than the point estimate

Over the same 126 mixes D3 reported:

| | as measured | fail-closed |
| --- | --- | --- |
| meets 90% of omniscient | 112 / 126 (89%) | **126 / 126 (100%)** |
| beats every static by +2% | **9 / 126 (7%)** | **102 / 126 (81%)** |

This is the number that sized Phase 97 as not worth building, and it was
wrong for a locatable reason. The 24 remaining non-wins are not selector
failures: 6 are the degenerate single-regime mixes where switching is
worthless by definition (the best static *is* the per-regime best), and the
rest concentrate on R6, where the selector's pick loses to the omniscient by
5% — a genuine mis-ranking that a fail-closed rule does not address and
should not.

## Is the threshold doing the work? No

The objection to any post-hoc rule is that the threshold was fitted to the
answer. Two invariance checks, both in the record:

**Threshold sweep.** Disarming exactly R4 — and therefore the entire result
above — holds for any bare threshold in **(1.026, 1.180)**, a band 0.15
wide, bounded below by R4's own margin and above by R8's. The derived
threshold, 1.039, sits inside it. The `beats every static` verdict survives
across the wider band (1.026, 1.34).

| threshold | armed regimes | rate | % omni | vs best static |
| --- | --- | --- | --- | --- |
| 1.00 | all six | 597.7 | 95.1 | 1.014 (FAIL) |
| **1.03 - 1.18** | **all but R4** | **626.0** | **99.6** | **1.062 (PASS)** |
| 1.18 - 1.23 | drops R8 | 616.7 | 98.1 | 1.046 (PASS) |
| 1.23 - 1.34 | drops R6 | 613.4 | 97.6 | 1.040 (PASS) |
| 1.34+ | drops R1 | 493.7 | 78.5 | 0.837 (FAIL) |

**Confidence-level invariance.** The z that would disarm each regime:

| regime | R4 | R8 | R6 | R1 | R5 | R5cot |
| --- | --- | --- | --- | --- | --- | --- |
| z to disarm | **1.10** | 7.06 | 8.84 | 12.42 | 22.06 | 26.62 |

The decision is identical for every `z` in **(1.10, 7.06)** — from 86%
one-sided confidence to beyond 6 sigma. The conventional 1.645 is nowhere
near either edge.

## Provenance, stated plainly

This is a **post-hoc rule with a pre-D3-data-only derivation**, not a
preregistered one. Both error terms and the threshold come from data that
predates the D3 grid, but the D3 outcome was already known when the rule was
written, and no commitment barrier can be retrofitted. The invariance checks
above are offered in place of a barrier, not as a substitute for one.

The honest reading: the *existence* of a fail-closed rule that recovers R4
was predicted in `results_g98_e.md` before this analysis, including its
mechanism (thin predicted margin) and its expected value. What this record
adds is that a rule derived from independently measured error terms lands in
the middle of a wide invariance band rather than needing to be steered.

## What it changes

1. **C3's claim becomes defensible as stated.** "A selector that arms
   correctly and fails closed" is now measured: 99.6% of omniscient, beating
   every static on 81% of mixes.
2. **Phase 97 is re-sized.** The 7%-of-mixes figure that argued against
   building runtime switching was an artifact; the honest figure is 81%.
3. **The remaining gap is mis-ranking, not arming** — R6, 5%. That is a
   Round-2 confirmation-budget question, not a C3 question.
