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

---

## 7. The batch drain term

Section 6 left a 1.5–2x over-prediction and named its cause: a constant-batch
model. The LO run's actual generation lengths, per request, are

```text
4721  5743  10603  12154  15546  24408  28196  32768
```

a **7x spread**, so the concurrent batch decays from 8 to 1 across the run and
its **mean active value is 4.09 of 8**. The run spends half its time at less
than half the nominal batch, and a step's batch-shared work is amortised over
whatever is left.

The term is physical rather than fitted. Per decode step, weights are read
**once** regardless of how many requests are in flight — as are the per-layer
launch constant, the window overhead and the floor — while KV is read **per
sequence**. So cost splits into a batch-shared part and a batch-proportional
part, and the active count is just the survival function of the
generation-length distribution, which every record already carries.

```text
per-token time = integral [ (verify_shared + D_shared)
                          + B(u) * (verify_per + D_per(u)) ] / tau(u) du
                 / integral B(u) du            with B(u) = #{i : G_i > u}
```

**Against the LO measurement**, with the verify split calibrated on the OFF
arm alone (residual +0.00% by construction) and everything else predicted:

| arm | no drain | **with drain** | measured |
| --- | --- | --- | --- |
| `woff/skip0` | 1.563x | **1.040x** | 1.034x |
| `woff/skip4` | 1.560x | **1.066x** | 1.023x |
| `woff/skip8` | 1.490x | **0.936x** | 0.897x |
| `w512/skip4` | 1.554x | **1.250x** | 1.408x |
| **mean relative error** | 0.450 | **0.051** | — |

**Mean error falls 9x, and three of four arms land within 4%** — including
the sign of the deep-skip arm, which the model now correctly places *below*
no-speculation. The drain was the dominant missing term, as section 6
predicted it would be.

The residual is one arm: `w512/skip4` is under-predicted by 11%, the window
saving more than the model credits it for. The KV-bytes coefficient is
inherited from an R5cot fit at a fixed 14K context, and a 512-token window
changes more than bytes read — attention work per step falls with it too. So
the next refinement, if one is wanted, is a window-aware attention term rather
than anything to do with batch.

## 8. Where the three terms leave the model

| model | mean relative error on LO |
| --- | --- |
| flat context, scalar tau | 1.015 |
| + context-integrated cost | 0.739 |
| + u-resolved acceptance | (folded into the above) |
| + batch drain | **0.051** |

All three terms were needed and they were needed in that order of magnitude:
the drain dominates, context growth is second, and acceptance's u-dependence
matters for *which* arm wins rather than for the level. None of them is
visible in a short-generation regime, which is why the R-regimes never
exposed them and the refined grid did.

---

## 9. The window-aware attention term: implemented, measured, and refuted

Section 7 left `w512/skip4` under-predicted by 11% and proposed a
window-aware attention term. Two things had to be faced before fitting one.

**A term scaling with KV positions is not identifiable.** Attention bytes and
attention flops are both linear in positions, so such a term is exactly
collinear with the KV-bytes term already in the model. The window can only
save *more* than a linear model credits if cost grows **superlinearly** with
positions — physically ordinary, since a 512-position working set is cache
resident while a 17000-position one streams from HBM. That is one parameter,
with the current model as its `gamma = 1` case:

```text
kv_cost(p) = kappa_kv * bytes(p) * (p / p_ref) ** (gamma - 1)
```

**One residual cannot identify a curve.** So instead of fitting gamma to the
single arm that motivated it, the shape was measured: a five-point window
sweep at fixed skip on the LO cell, plus acceptance curves for every window.

### What the sweep measured

| window | per-token | vs off | tau at u<256 | tau at 3K–8K |
| --- | --- | --- | --- | --- |
| off | 2.803 ms | 1.023x | 4.553 | 4.709 |
| 128 | 2.666 ms | 1.076x | 4.057 | **3.772** |
| 256 | 2.641 ms | 1.086x | 4.440 | 3.922 |
| 512 | 2.037 ms | 1.408x | 4.553 | 4.266 |
| **1024** | **1.943 ms** | **1.476x** | 4.583 | 4.346 |

**Throughput is not monotone in window size.** A KV-bytes model says tighter
windows are strictly cheaper, so w128 should win; it measures 27% *worse*
than w1024. The acceptance columns say why: acceptance falls monotonically as
the window tightens, and its decay with generation length steepens — by the
3K–8K bucket the penalty against no window is 7.7% at w1024, 9.4% at w512,
16.7% at w256 and 19.9% at w128.

So **the window lever has an interior optimum**, at 1024 or beyond for this
cell, and it is invisible to any model that prices windows by bytes alone.

### The verdict on gamma

Refitted over the five arms with each window's own measured acceptance:

| gamma | 1.00 | 1.10 | 1.20 | 1.30 | 1.40 | 1.50 |
| --- | --- | --- | --- | --- | --- | --- |
| mean error | **0.1209** | 0.1216 | 0.1220 | 0.1222 | 0.1223 | 0.1222 |

**gamma = 1 is best, and raising it monotonically hurts.** The data does not
support a superlinear KV term; the residual is not attention non-linearity.
The term ships with `gamma = 1.0` as its default — a no-op that reproduces
the existing model exactly — and the record says it was measured and not
adopted rather than left as an untested option.

The remaining ~12% is most likely in the cost coefficients themselves:
`kappa_kv` is inherited from an R5cot fit taken at a fixed 14K context and
batch 8, and nothing here re-fits it on the refined cells. That, not another
functional form, is what would close the gap.

**A bug found in my own fitting**: the first gamma sweep returned an
identical error for every value, which is impossible if the parameter is
live. `integrated_with_drain` reaches cost through `split_cost`, not
`draft_cost`, so gamma never entered the computation. The insensitivity is
what exposed it; the numbers above come from the corrected path.

---

## 10. Refitting `kappa_kv` on the refined cell: done, and not adopted

Section 9 attributed the residual to `kappa_kv` carrying the wrong regime —
inherited from an R5cot fit at batch 8 and a fixed 14K context, applied to a
cell whose context sweeps 120 to 33000 under a decaying batch. The refined
cell can identify the coefficients itself, because one arm's measured time is
linear in them:

```text
time = (v_shared + F)*S + keep*A*S + keep*f_win*[w>0]*S
       + keep*kv_bytes*kappa_kv*R + v_per*Q
```

Seven armed arms, four unknowns. The design varies `keep` over
{1.0, 0.889, 0.778} — which separates the per-layer term from the floor — and
the window over {off, 128, 256, 512, 1024}, which is what should pin
`kappa_kv`. (`A` absorbs the weight term: every arm shares the target's
weights, so `W*kappa_w` is constant across this arm set and collinear with
the per-layer constant.)

### The fit is degenerate unconstrained

Plain least squares returns **`kappa_kv` = −1.12e−11, `F` = −10.8 ms,
`f_win` = −57 ms**: a negative floor, a window that pays to exist, and KV
that makes the draft *faster* the more of it you read. Physically impossible,
and the reason is in the design:

| | F | A | f_win | kappa_kv |
| --- | --- | --- | --- | --- |
| f_win | +0.287 | +0.358 | +1.000 | **−0.985** |

**`f_win` and `kappa_kv` are collinear at −0.985.** Every windowed arm carries
the indicator, so the only thing separating the two columns is how window
size moves KV exposure — and least squares happily trades a large negative KV
coefficient against a large negative window constant.

### Under a non-negativity constraint

| parameter | refit (LO) | inherited (R5cot) |
| --- | --- | --- |
| `kappa_kv` | **8.08e−13** | 8.62e−12 (**10.7x larger**) |
| `F` | 9.61 ms | — |
| `A` (per-layer, keep=1) | 22.4 ms | — |
| `f_win` | 0 (at the boundary) | — |
| mean residual | **0.098** | 0.121 |

The refit says the inherited KV coefficient is **an order of magnitude too
large** for this cell, and fitting on the cell improves the residual from
0.121 to 0.098.

### Why it is not adopted

Two defects, both in the data rather than the arithmetic:

1. **`f_win` is pinned to zero by the constraint, not by measurement.** With
   a −0.985 correlation the split between the window constant and the KV
   coefficient is not identified; NNLS resolves it at a boundary, which is an
   artifact. Breaking it needs unwindowed-vs-windowed contrasts at more than
   one keep — the current sweep varies window only at skip4.
2. **Each arm ran a different workload.** Realized output spans 104K to 159K
   tokens, a **1.52x spread**, because natural EOS plus T=0 divergence gives
   every arm its own generation lengths. The regressors therefore encode
   workload as well as configuration.

Both are fixed by the same thing, and it is a protocol rather than a model:
**cost must be calibrated under equal work.** That is exactly what Phase 98's
`ignore_eos` instrument provides and exactly what Phase 100 retires from
*scoring* while keeping for *measurement*. The two protocols are for
different purposes, and this attempt demonstrates the boundary by failing
across it: natural EOS is right for scoring throughput and wrong for fitting
coefficients.

So the number stands as a measured indication — the inherited `kappa_kv` is
roughly 10x too large at LO — and not as a calibration. The calibration run
is an equal-work sweep on the refined cells with the window varied at two or
more keeps, which is a well-specified next campaign rather than a further
model refinement.
