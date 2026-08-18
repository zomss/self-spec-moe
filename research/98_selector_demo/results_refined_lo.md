# tau(u) in the prediction, and the LO cell measured under the refined protocol

Two deliverables: the u-resolved acceptance estimator (`w98_tau_u`), and the
first measurement of our arms on the refined grid's long-output cell under
Phase 100's protocol. The second contradicts the natural reading of the
first, which is the useful part.

## 1. tau_eff: the right scalar to feed a throughput prediction

Over a generation of `G` tokens the quantity that integrates is **steps per
token**, so the correct summary of a tau curve is the token-weighted
*harmonic* mean:

```text
tau_eff(G) = G / sum_b (tokens_b / tau_b)
```

An arithmetic mean would overstate throughput by letting high-acceptance
stretches pay for low-acceptance ones at face value, when the slow stretches
in fact consume disproportionately many steps. Unmeasured tail buckets carry
the last measured bucket forward, which under-states both measured trends
rather than inventing their continuation. 13 tests, including the two real
curves.

Applied to the measured curves, at the LO cell's actual generation length
(~13-17K tokens) against the old sub-1K calibration:

| arm | tau (calibration) | tau_eff (LO length) | shift |
| --- | --- | --- | --- |
| `woff/skip0` | 4.945 | 4.969 | +0.5% |
| `woff/skip4` | 4.555 | 4.681 | +2.8% |
| `woff/skip8` | 3.677 | 4.051 | **+10.2%** |
| `w512/skip4` | 4.458 | 4.272 | **−4.2%** |

## 2. The LO cell, measured

Phase-100 protocol: LO prompts from the registered bundle, **natural EOS**,
32K cap, batch 8, five arms. Scored in Campaign 1's currency — per-token cost
against the parked arm, with amendment 1's context correction — because the
arms do not emit identical outputs (section 4).

| arm | tok/s | out tokens | per-token | **vs OFF (corrected)** |
| --- | --- | --- | --- | --- |
| `w512/skip4` | 491.0 | 127,838 | 2.037 ms | **1.421x** |
| `woff/skip0` | 360.5 | 113,537 | 2.774 ms | 1.052x |
| `woff/skip4` | 356.8 | 113,730 | 2.803 ms | 1.042x |
| OFF | 348.6 | 134,139 | 2.868 ms | 1.000x |
| `woff/skip8` | 312.7 | 104,530 | 3.198 ms | **0.914x** |

**The windowed arm wins by 42%; the unwindowed arms barely clear parity; deep
skip loses outright.** The direction agrees with Campaign 1, which found the
window family taking LO on h103/h104.

## 3. What this says about tau(u) — including against it

The naive expectation from section 1 was that u-resolved acceptance would
sharpen the LO prediction. On the acceptance axis alone it points the **wrong
way**: tau_eff makes the winning arm look *worse* (−4.2%) and the losing arm
look *better* (+10.2%). Acceptance alone therefore misranks this cell.

The reason is that **LO is cost-dominated**. At a 17K context a 512-token
window reads ~30x less KV per draft step, and that saving swamps the ~9%
acceptance penalty the window pays by 8K of generation. So the two measured
effects push against each other and cost wins:

* window, acceptance side: **−9.4%** by 8K (`results_g98_longu.md`)
* window, cost side: enough to produce **+42%** end to end

A consistency check — inverting each arm's per-token time to infer its draft
cost, using the OFF arm's 2.868 ms as a direct measurement of `verify` —
does **not** cleanly separate the two acceptance models, and is reported as
inconclusive: the LO runs use the live K/OFF ladder, which may park steps, so
"per-token time x tau" is not a clean inversion. Under it the scalar happens
to give more structurally plausible skip-scaling (0.820 against an expected
~0.85 with the floor) than tau_eff (0.924), but the confound is large enough
that this does not adjudicate.

**Conclusion for the selector**: tau(u) is necessary but not sufficient. The
prediction's larger error at LO is on the *cost* side, because the current
model consumes one context per regime while an LO generation grows its
context from ~120 tokens to ~17K. A context-growth-aware cost term is the
next piece, and it is what would actually let the selector see the window's
advantage here.

## 4. Two protocol facts found by running it

**Geometry.** Our engine defaults to `max_model_len` 20480 while LO's cap is
32768. Long generations ran into the context limit, the scheduler shortened
the draft near it, and the K/OFF registry rejected the resulting width
("permits only K=0 or K=4, got K=2"). Phase 100 re-pins the geometry to
40960 for exactly this reason. Before the fix the parked arm silently
generated 102K tokens; after it, 134K — the earlier run was truncating.

**T=0 outputs are not identical across arms.** The refined design argues
equal work is preserved by losslessness, so all arms stop at the same EOS.
Measured here, output totals span 104K-134K (28%). Campaign 1's own records
show the same thing and are explicit about it: every armed arm diverges from
stock on 16/16 LI/LIO/LO prompts. This is the Phase-96 W9 mechanism —
verification runs the target at different batch shapes, FP reduction order
changes, near-tie argmaxes diverge, and EOS lands elsewhere. Campaign 1
already handles it by scoring per-token with a context correction rather than
by raw throughput, and by rescoping the identity gate to a diagnostic
(amendment 1). This run adopts that currency; raw tok/s would have ranked
the arms differently and wrongly.

## 5. Scope

* One cell (LO), one batch (8), one box, single boots per arm. The ranking
  gap between first and last is 55%, far outside this box's ~1% armed-arm
  stability, but the 1.042 / 1.052 pair is inside it and is not separated.
* Acceptance curves come from 8192-token generations while these runs average
  13-17K, so tau_eff extrapolates by carry-forward over the last third.
* `w512/skip4` is our arm, not Campaign 1's `magicdec512` (`w512/skip0`), and
  our `skip8` is the registered set, not their `knap8`. The numbers are
  therefore comparable in kind, not cell for cell.

---

## 6. The context-growth-aware cost term (`w98_cost_u`)

Section 3 located the remaining LO error on the cost side. The term is now
implemented: draft cost is integrated over the generation instead of
evaluated at one context,

```text
per-token time = (1/G) * integral_0^G [verify(p+u) + D(p+u)] / tau(u) du / B
```

with `D(p+u)` the Round-2 model re-evaluated as the context grows, `tau(u)`
from `w98_tau_u`, and `B` the concurrent batch. Nine tests, including the two
structural properties that motivated it: an unwindowed draft's KV term grows
by exactly `kappa_kv * delta_context`, while a windowed draft's **saturates**
at `window + sinks` — and the integrated model therefore favours the window
by more than a flat-context model does, which is the whole point.

**Applied to the LO measurement** (R5cot fit, verify shape from the batch-8
regimes with its level calibrated from the OFF arm):

| arm | flat pred | integrated pred | measured |
| --- | --- | --- | --- |
| `woff/skip0` | 2.262x | 1.843x | 1.034x |
| `woff/skip4` | 2.216x | 1.858x | 1.023x |
| `woff/skip8` | 1.911x | 1.754x | 0.897x |
| `w512/skip4` | 2.219x | **1.977x** | **1.408x** |
| mean relative error | 1.015 | **0.739** | — |
| rank correlation | +0.800 | +0.800 | — |

The integrated term cuts absolute error by **27%** and both models pick the
right winner. Ranking is unchanged, because at LO the window's advantage is
large enough that either model finds it.

**Both still over-predict by roughly 1.5–2x, and the residual has a named
cause.** Predictions assume a constant batch of 8; under natural EOS requests
finish at different lengths, so the effective batch decays through the run and
per-token cost rises above what a constant-batch model gives. That is the
Phase-96 W9 drain mechanism appearing on the cost side, and it is the next
term to add — not another context refinement.

**One error found in my own analysis and corrected here**: the first pass
omitted the batch divisor entirely, predicting armed arms 2–3x *slower* than
OFF and producing an unphysical negative verify intercept. A per-step model
compared against per-token measurements is a units error, and the negative
intercept is what exposed it.
