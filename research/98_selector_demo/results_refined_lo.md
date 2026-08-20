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

---

## 11. The equal-work calibration — and what it overturns

Section 10 specified the fix: calibrate cost under **equal work**, with the
window varied at more than one keep. Run here as 10 arms (3 windows x 3
skips, plus `w512/skip4`), `ignore_eos` with a fixed 8192-token budget at
batch 8 on LO content, and the draft chain read **from the profiler** rather
than inverted out of throughput — so acceptance leaves the calibration
entirely.

Equal work is visible in the data: every arm emitted exactly 65,536 tokens,
differing only in how many steps it took (1661 for `woff/skip0`, 2406 for
`w256/skip8`).

### The fit

| parameter | **equal-work (LO)** | inherited (R5cot) | natural-EOS refit |
| --- | --- | --- | --- |
| `kappa_kv` | **1.035e−11** | 8.622e−12 | 8.082e−13 |
| `f_win` | **1.510 ms** | −1.192 ms | 0 (boundary) |
| `A` per-layer | **26.356 ms** | — | 22.425 ms |
| `F` floor | **3.791 ms** | 3.362 ms | 9.612 ms |
| **mean residual** | **0.0035** | 0.1209 | 0.0984 |
| physical unconstrained | **yes** | — | **no** |

**Mean residual 0.35%**, every arm within ±0.7%, and — the decisive check —
the unconstrained solution is **physically admissible on its own**, with NNLS
agreeing to within 5%. The negative KV coefficient that made section 10's fit
degenerate is gone.

The keep axis validates itself: measured draft chain falls 36.844 → 32.933 →
29.135 ms across skip 0/4/8, ratios of 0.894 and 0.791 against keep fractions
of 0.889 and 0.778.

### What it overturns

Section 10 concluded from the natural-EOS data that the inherited `kappa_kv`
was **~10x too large**. That conclusion was wrong, and it was flagged as
not-adoptable at the time for exactly the reasons that turn out to matter.
Under equal work `kappa_kv` is **1.20x the inherited value** — a modest
correction in the *opposite direction*.

The lesson is the protocol, not the number. A degenerate fit on confounded
data produced a confident order-of-magnitude claim; the fix was not a better
estimator but a measurement where every arm does identical work. Section 10's
refusal to adopt its own result is what kept the error out of the model.

### Honest residual concerns

* `f_win` and `kappa_kv` still correlate at **−0.957** — better than −0.985
  but not comfortable. The fit is physical and the residual is 0.35%, so the
  identification held; a window sweep at more keeps would tighten it further.
* The profiler's ~2.4 ms/step syncs (X25) inflate every arm's draft chain
  equally and land in `F`, whose fitted 3.791 ms sits just above Round 1's
  registered 3.66 ms floor. The two are not independent measurements.
* One cell, one batch, one model. These coefficients are calibrated for LO's
  context regime and should not be transferred back to the R-regimes without
  the same check that motivated this run.

---

## 12. LO re-scored with the equal-work coefficients

The drain-integrated prediction, rerun over all seven armed LO arms with the
section-11 coefficients in place of the inherited R5cot ones. Each arm uses
its own measured acceptance curve and its own realized generation lengths;
the verify level is calibrated once on the parked arm.

| arm | inherited | **equal-work** | measured |
| --- | --- | --- | --- |
| `woff/skip0` | 1.415 | **1.018** | 1.034 |
| `woff/skip4` | 1.431 | **1.043** | 1.023 |
| `woff/skip8` | 1.239 | **0.917** | 0.897 |
| `w256/skip4` | 1.603 | **1.065** | 1.086 |
| `w1024/skip4` | 2.166 | **1.445** | 1.476 |
| `w512/skip4` | 1.767 | 1.176 | **1.408** |
| `w128/skip4` | 1.960 | 1.302 | **1.076** |
| **mean error** | **0.4524** | **0.0675** | — |

**Error falls 6.7x**, and five of seven arms land within 2–6% — including the
sign at `woff/skip8`, which the model correctly places below no speculation.

The earlier headline of 0.051 was over a four-arm subset that predates the
window sweep; on this seven-arm set the inherited coefficients score 0.452,
so the comparison above is the like-for-like one.

### Two residuals, and one of them is the measurement

`w512/skip4` is still under-predicted (1.176 against 1.408) and `w128/skip4`
is now over-predicted (1.302 against 1.076). The model orders the window
family `w1024 > w128 > w256 > w512`; measurement says
`w1024 > w512 > w256 > w128`.

The cost side cannot be responsible. In the equal-work data the draft chain
is monotone in window — 33.073 ms at w1024, 32.033 at w256 — and the modelled
KV difference between a 144-position and a 528-position window is ~0.07 ms
per request per step against a ~30 ms step. Among windows, cost is nearly
flat; acceptance is what should decide, and the measured curves have w128
11.6% below w512 by the 3K–8K bucket.

`w128/skip4` is also the workload outlier: its run emitted **159K tokens with
3 cap hits**, against 104–128K and 0–1 for every other arm, because its low
acceptance changed the sampled tokens and pushed three requests into the 32K
cap. Its predicted-versus-measured comparison is the least trustworthy of the
seven for that reason, and the honest reading is that the window ordering
below 512 is not yet resolved rather than that the model has it wrong.

### The sync correction does not behave

The calibration ran under the **legacy** instrument (profiler syncs present,
~2.4 ms/step per X25) while the scored throughput runs used **corrected**, so
the fitted floor should arguably be reduced by that amount. It should not be:
subtracting it moves the mean error the wrong way, 0.0675 to 0.0972. `F` is
therefore kept as fitted and the discrepancy recorded as open — either the
sync cost is not a clean per-step constant at this batch, or the corrected
arm still carries part of it. That is a question about the instrument, not
about the cost model.

---

## 13. The transfer test — and the term that was "refuted" comes back

The sharpest untested claim was whether coefficients fitted on ONE cell
predict others, since that is the search strategy's whole premise. LI and LIO
are the right test because their context profile is the **inverse** of LO's:
~12K prompts with 512-token generations, against LO's 120-token prompt
growing to 8K. Four arms each, equal work, same instrument.

### Applying LO's coefficients unchanged

| cell | arm | KV positions | measured | error |
| --- | --- | --- | --- | --- |
| LI | `w256/skip4` | 272 | 28.96 ms | **−0.1%** |
| LI | `w1024/skip4` | 1040 | 29.77 ms | **+0.7%** |
| LI | `woff/skip4` | 12416 | 48.58 ms | **−9.3%** |
| LI | `woff/skip0` | 12416 | 55.32 ms | **−11.2%** |
| LIO | `w256/skip4` | 272 | 28.83 ms | +0.3% |
| LIO | `w1024/skip4` | 1040 | 29.42 ms | +1.9% |
| LIO | `woff/skip4` | 9863 | 45.60 ms | −11.0% |
| LIO | `woff/skip0` | 9863 | 52.48 ms | −13.9% |

**Windowed arms transfer almost exactly; unwindowed arms under-predict by
9–14%.** The error is not diffuse — it lands entirely on the arms whose KV
sits far beyond the calibration range, and it is one-directional: real cost
grows **faster** than linear in KV positions.

That is the superlinear term section 9 implemented and reported as refuted.
The refutation was correct for the data it had and wrong as a general
conclusion, for a reason now visible: **LO spans 271–4252 KV positions, LI
spans 271–12416.** LO's lever arm was three times too short to identify a
curvature, so the fit sensibly returned `gamma = 1`.

### Refitting gamma across all three cells

| gamma | LO | LI | LIO | all |
| --- | --- | --- | --- | --- |
| 1.00 | **0.35%** | 5.33% | 6.76% | 2.88% |
| 1.20 | 0.76% | 1.26% | 3.74% | 1.53% |
| **1.25** | 0.88% | **0.89%** | **2.96%** | **1.34%** |
| 1.30 | 0.98% | 1.91% | 2.14% | 1.44% |

**gamma = 1.25 halves the error**, 2.88% to 1.34%, and does it in the way a
real effect does rather than a fitted one: it gives up a little on the cell
that calibrated it (0.35% to 0.88%) to gain a great deal on the two it never
saw (5.33% to 0.89%, 6.76% to 2.96%). KV cost scales as `p^1.25` — attention
gets less efficient per position as the working set grows, which is ordinary
memory-hierarchy behaviour and the same physics that makes windows pay.

### What this settles, and what it does not

**Settles**: the cost model transfers. With one curvature parameter it
predicts three cells spanning 271 to 12416 KV positions and two opposite
context profiles to **1.34% mean error**, having been fitted on one of them.
That is the search strategy's premise holding on the refined grid.

**Does not settle**: LIO's unwindowed arms remain 4–7% under-predicted at
`gamma = 1.25` while LI's are within 1.5%, so something separates the two
long-prompt cells that a single curvature does not capture — plausibly that
LIO's prompts are longer-tailed, making the mean KV position a worse summary
of a run whose requests differ more in length. Retiring that would need
per-request rather than per-run integration.

**Method note.** Section 9's refutation is left standing in the record with
this section appended rather than rewritten. It was right about its data and
wrong about the world, and the distinction — a parameter that is
unidentifiable in one design and measurable in another — is the reusable
part.

---

## 14. The quant axis: `kappa_w` identified, and the additive model refuted

Six `w4a16-quantized` arms (windows `off/256/1024` x keeps `1.0/0.889`)
measured under the same equal-work protocol, on a box gated to reproduce
`target-matching/woff/skip4` to **0.02%** (32.941 ms against the 32.933 ms
quiet-box reference) before any arm was allowed to run.

They were added for one reason: every arm measured until now shared the
target's weights, so `W * kappa_w` was **constant across the design** and
perfectly collinear with the per-layer constant. The fit could only ever
report their sum, `A`. A second weight version — 13.892 GB of body bytes
against 3.581 GB — is the only thing that separates them.

### It works: the collinearity breaks

| coefficient | 10 arms, one weight version | 16 arms, two |
| --- | --- | --- |
| `A` (per-layer, keep-scaled) | 26.36 ms | — |
| `kappa_w` | **unidentifiable** | 8.769e-13 s/byte |
| `c_layer` | **unidentifiable** | 17.18 ms |
| `kappa_kv` | 1.035e-11 | 4.264e-12 |
| `f_win` | +1.51 ms | **−1.28 ms** |
| condition number | 44.5 | 47.6 |
| mean residual | **0.35%** | 1.23% |
| physical / NNLS agrees | yes / yes | **no / no** |

Weight traffic is **12.18 ms of the bf16 draft chain and 3.14 ms of the
quantized one** — 41.5% of `A` against 15.5%. That is the number the phase
has never had, and it says the quantized draft's step is no longer
weight-dominated: it is mostly `c_layer`, the per-layer cost that
quantization does not touch.

### And it refutes the model that produced it

Three things go wrong at once, and they are one thing. `f_win` turns
**negative**, the unconstrained fit stops being physical, NNLS stops
agreeing, and the residual rises 3.5x. None of that happened with ten arms.

The cause is visible without any model. For every `(window, keep)` measured
under **both** weight versions — pairs differing in nothing but the draft's
weights, sharing the target's KV cache either way:

| window | keep | bf16 | w4a16 | quant saves | implied `kappa_w` |
| --- | --- | --- | --- | --- | --- |
| off | 1.000 | 36.844 | 26.659 | **10.186 ms** | 9.879e-13 |
| off | 0.889 | 32.933 | 24.401 | 8.532 ms | 9.308e-13 |
| 1024 | 1.000 | 33.073 | 23.513 | 9.560 ms | 9.272e-13 |
| 1024 | 0.889 | 29.947 | 21.657 | 8.289 ms | 9.043e-13 |
| 256 | 1.000 | 32.033 | 24.026 | **8.008 ms** | 7.766e-13 |
| 256 | 0.889 | 28.897 | 21.976 | 6.920 ms | 7.550e-13 |

**Quantization is worth 21% less at a 256-token window than unwindowed**
(10.186 → 8.008 ms), and the same at the other keep (8.532 → 6.920, −19%).
The implied per-byte coefficient spreads **30.8%** and is ordered by window,
not scattered. If quantization were simply fewer weight bytes, the window
could not matter — the bytes are the same either way.

The same interaction, seen from the other side, is the sharper statement:

| draft | w256 → w1024 (+721 KV positions) |
| --- | --- |
| `target-matching` | **+1.040 ms** (tighter window saves) |
| `w4a16-quantized` | **−0.513 ms** (tighter window *costs*) |

Same shared KV cache, same positions, **opposite sign**. For the quantized
draft, `w256` is more expensive than `w1024` at both keeps (24.026 vs 23.513;
21.976 vs 21.657) — a window inversion, at 1.3–2x this box's armed-arm
stability band, reproduced across two independent arms.

### A bandwidth account rules out the innocent explanation

If the two weight versions differed only in bytes read, at a shared effective
bandwidth, then with `chain = W_bytes/BW + other` and `other` common:

```text
36.844 ms = 4 x 13.892 GB / BW + other      (K=4 draft forwards per chain)
26.659 ms = 4 x  3.581 GB / BW + other
---------------------------------------
10.185 ms = 41.25 GB / BW   =>   BW = 4050 GB/s
```

The box is an **H100 80GB HBM3: 3350 GB/s peak**. The implied rate is 1.21x
peak, so the premise is false — `other` is *not* common. The quantized path
changes the kernels (Marlin GEMMs replace cuBLAS), not merely the byte count,
and a single byte column charges the fit for all of it.

### What this settles, and what it costs

**Settles**: `kappa_w` is identified, and the quantized draft is only ~12% weight
traffic. Skipping layers and quantizing therefore compete for the same
shrinking share of the step, which is the cost-side reason deep skip stops
paying once quantization is on — previously an inference, now a measurement.

**Costs**: the additive cost model `D = keep*(W*kw + KV*kkv + c_layer +
f_win) + F` is **refuted as a joint model over both weight versions**. It
holds within a weight version (0.35% over ten bf16 arms) and fails across
them (1.23%, non-physical). Levers do not add; quant and window interact,
and in the direction that matters — each is worth less when the other is on.

The leading mechanism is saturation: unwindowed bf16 is the most
bandwidth-bound arm, so cutting weight bytes pays most there; the quantized
windowed arm is already off the bandwidth wall, so cutting KV positions
further buys nothing and the tighter window's fixed path cost dominates. It
predicts that the inversion should weaken as batch rises, which is the test
and is not run.

**Not adopted**: the 16-arm coefficients are recorded, not shipped. A fit that
returns a negative window cost would mispredict any window it has not seen,
and the LO re-scoring of section 12 keeps the ten-arm bf16 coefficients it was
built on.

**A gap this exposes.** Sections 2–13 measure `target-matching` arms
exclusively, while every selector pick in the 31-cell grid was `w4a16`. The
refined-grid evaluation therefore has not yet scored the arm family the
selector actually chooses, and this section is the first evidence that the
two families do not share a cost surface.

---

## 15. The w4a16 family on the refined grid: the lattice reorders

Section 14 closed on a gap: sections 2–13 scored `target-matching` arms
exclusively while every selector pick in the 31-cell grid was `w4a16`, and
the calibration had just shown the two families do not share a cost surface.
This closes it — the same eight configurations, same protocol, same prompts,
same cell, under the other draft weight version.

### The denominator checks out first

Anything scored against OFF on this box carries a 9–16% band, so two families
scored against two separate parked boots are only comparable if those boots
reproduced. They did, to a degree the band did not promise:

| | `target-matching` run | `w4a16` run |
| --- | --- | --- |
| OFF tok/s | 348.6 | 348.4 |
| OFF output tokens | 134,139 | **134,139** |
| OFF per-token | 2.868 ms | 2.870 ms (**0.07%**) |

An identical token count is the stronger half: the parked path is genuinely
quant-independent — the draft is resident but never runs — so this is one
measurement made twice, and the two families sit in one table by right.

**A confound retires here too.** Section 3 could not invert measured time by
acceptance because "the LO runs use the live K/OFF ladder, which may park
steps". The traces say it never did: **armed fraction is 99.97% in all
fourteen armed runs.** These are pure armed measurements.

### The result

| arm | `target-matching` | `w4a16` | quant buys | acc bf16 | acc w4a16 |
| --- | --- | --- | --- | --- | --- |
| `w1024/skip4` | **1.496** | **1.565** | +4.6% | 0.836 | 0.836 |
| `w512/skip4` | 1.421 | 1.509 | +6.1% | 0.841 | 0.817 |
| `w256/skip4` | 1.113 | 1.332 | +19.7% | 0.740 | 0.753 |
| `woff/skip0` | 1.052 | 1.449 | +37.6% | 0.992 | 0.939 |
| `w128/skip4` | 1.051 | 1.301 | +23.8% | 0.740 | 0.696 |
| `woff/skip4` | 1.042 | 1.443 | +38.4% | 0.922 | 0.878 |
| `woff/skip8` | **0.914** | **1.562** | **+70.8%** | 0.762 | 0.800 |

**Quantization wins every arm, by between 4.6% and 70.8%** — and the size of
the win is ordered inversely by how tight the window is, which is section
14's interaction seen end to end rather than in a draft-chain profile.

**The lattice reorders.** Spearman between the two families' rankings is
**+0.286** (p=0.535, n=7): statistically indistinguishable from no
relationship. The R-grid's content pair R5 vs R5cot — different task, same
everything else — correlated at **+0.902**. So on this cell, **changing the
draft's weight version reorders the lever lattice far more than changing the
workload does**, and "pick the window and skip, then quantize" is not a
decomposition the data supports.

**`skip8` flips from worst to near-best**: the only sub-parity arm in the
bf16 family (0.914, an 8.6% regression) becomes the second-best quantized arm
(1.562). Same skip set, same cell, same prompts; only the draft's weights
differ.

### Why, from the traces

Acceptance is nearly family-independent — within 6% at every one of the seven
configurations — so the family gap is a cost effect, with one exception, and
the exception is the flip:

| | bf16 | w4a16 |
| --- | --- | --- |
| acceptance, `skip0` → `skip8` | 0.992 → 0.762 (**−23.2%**) | 0.939 → 0.800 (**−14.8%**) |
| draft chain, `skip0` → `skip8` | 36.844 → 29.135 ms (−20.9%) | 26.659 → ~22.1 ms (−17%, extrapolated) |

bf16 pays 23.2% of its acceptance to buy 20.9% of its draft cost — a losing
trade, and measured as one. The quantized draft pays 14.8% to buy ~17% — a
winning trade. The asymmetry has a plain cause: `target-matching/woff/skip0`
is the **degenerate arm**, a draft whose weights are the target's own tensors
with nothing removed, accepting 0.992 of what it drafts. It sits on the
acceptance ceiling, so every lever applied to it can only fall off, and falls
fast. The quantized draft starts at 0.939, already off the ceiling, and the
same eight skipped layers cost it markedly less.

Only the `skip0 → skip8` quantized cost is extrapolated: the equal-work sweep
measured keeps 1.0 and 0.889 under w4a16, not 0.778. The end-to-end number it
supports is measured, not extrapolated.

### What this changes

**Good news for the headline.** The best arm on this cell rises from 1.496 to
**1.565**, and the best pick is `w1024/skip4` in *both* families — the top of
the lattice transfers even though the ranking does not. That is why the
31-cell selector scored 0.982 while never having been checked across weight
versions: it was right where it mattered, not right throughout.

**A live risk in the design, now concrete.** Round 1 eliminates candidates by
predicted cost, from coefficients fitted on one weight version. Every
coefficient this phase owns was fitted on `target-matching` arms. On bf16
evidence `skip8` is the worst arm in the lattice and the obvious thing to
eliminate; in the quantized lattice it is second best at 1.562. **The
elimination round can discard the runner-up of the family the selector
actually deploys.** Nothing in the current design would notice.

**Scope.** One cell, one batch, single boots. The 1.496/1.421 and 1.052/1.051
pairs sit inside this box's ~1% armed-arm stability and are not separated;
the flip (0.914 → 1.562) and the family gaps at `woff` (+37–71%) are one to
two orders outside it. `w128/skip4` remains the outlier flagged in section 12
(159K tokens, 3 cap hits) in the bf16 family; its quantized twin emitted 125K
with 1, so the bf16 w128 number is the less trustworthy of that pair.

**Still open**: no u-resolved acceptance curve exists for the quantized
family — `results_g98_longu.md` measured `target-matching` only — so these
arms can be measured but not yet predicted. That is the next input the
selector needs on this grid.

---

## 16. The quantized family's u-curves: the missing input, measured

Section 15 closed on the one thing blocking prediction: every acceptance
curve the phase owned was `target-matching`, so the quantized arms could be
measured but not predicted. All seven are now measured under the same
instrument — LO content, 8192 tokens, `ignore_eos`, u-edges
`[256, 1024, 3072, 8192]`, depth 8.

### tau by bucket, both families

| arm | u<256 | 256–1K | 1K–3K | 3K–8K | drift bf16 / w4a16 |
| --- | --- | --- | --- | --- | --- |
| `woff/skip0` | 8.851 / 7.398 | 8.757 / 7.465 | 8.822 / 7.880 | 8.891 / 8.045 | +0.4% / **+8.7%** |
| `woff/skip4` | 7.270 / 6.367 | 7.386 / 6.864 | 7.527 / 7.388 | 7.918 / 7.672 | +8.9% / **+20.5%** |
| `woff/skip8` | 5.012 / 4.916 | 5.122 / 4.960 | 5.782 / 5.688 | 6.257 / 5.972 | +24.8% / +21.5% |
| `w128/skip4` | 6.186 / 5.742 | 5.679 / 5.385 | 5.572 / 5.205 | 5.585 / 5.064 | −9.7% / −11.8% |
| `w256/skip4` | 7.027 / 6.267 | 6.277 / 6.025 | 5.968 / 5.526 | 5.841 / 5.497 | −16.9% / −12.3% |
| `w512/skip4` | 7.270 / 6.367 | 6.968 / 6.568 | 6.597 / 6.673 | 6.722 / 6.348 | −7.5% / **−0.3%** |
| `w1024/skip4` | 7.279 / 6.367 | 7.365 / 6.848 | 6.929 / 6.785 | 6.969 / 6.609 | −4.3% / **+3.8%** |

The sign of the u-trend is family-independent — unwindowed arms gain with
generation length, windowed arms lose — which is the reassuring half.

### What quantization costs in acceptance, and where

| arm | u<256 | 256–1K | 1K–3K | 3K–8K |
| --- | --- | --- | --- | --- |
| `woff/skip0` | **−16.4%** | −14.8% | −10.7% | **−9.5%** |
| `woff/skip4` | −12.4% | −7.1% | −1.9% | −3.1% |
| `woff/skip8` | **−1.9%** | −3.1% | −1.6% | **−4.6%** |
| `w128/skip4` | −7.2% | −5.2% | −6.6% | −9.3% |
| `w256/skip4` | −10.8% | −4.0% | −7.4% | −5.9% |
| `w512/skip4` | −12.4% | −5.7% | +1.2% | −5.6% |
| `w1024/skip4` | −12.5% | −7.0% | −2.1% | −5.2% |

Two results, and both bear on section 15's flip.

**Quantization's acceptance penalty shrinks as generation lengthens.** At
`skip4` it falls from −12.4% to −3.1%. G98-D measured acceptance over 640
generated tokens — entirely inside the first two buckets — so **the
calibration the selector inherited overstates the cost of quantizing by two
to four times for the regime the LO cell actually occupies.**

**Deep skip is nearly free to quantize.** `skip8` pays 1.6–4.6% at every
bucket while `skip0` pays 9.5–16.4%. This is the acceptance half of the flip
section 15 measured on the cost side, and it is the same ceiling argument:
`target-matching/woff/skip0` accepts 0.99 of what it drafts, so any
degradation shows; a draft already down to tau 5 has little left to lose.
Quantization and skip damage the *same* thing, so their damages do not add —
just as their savings do not.

### Why this is the input the selector was missing

The selector consumes one scalar tau per cell, pooled below 1K. Against what
an LO generation actually realizes:

| arm | bf16 scalar → eff | error | w4a16 scalar → eff | error | measured (bf16/w4a16) |
| --- | --- | --- | --- | --- | --- |
| `woff/skip0` | 8.804 → 8.873 | −0.8% | 7.432 → 7.983 | −6.9% | 1.052 / 1.449 |
| `woff/skip4` | 7.328 → 7.816 | −6.2% | 6.615 → 7.569 | **−12.6%** | 1.042 / 1.443 |
| `woff/skip8` | 5.067 → 6.070 | **−16.5%** | 4.938 → 5.850 | **−15.6%** | 0.914 / 1.562 |
| `w128/skip4` | 5.933 → 5.594 | +6.1% | 5.563 → 5.107 | +8.9% | 1.051 / 1.301 |
| `w256/skip4` | 6.652 → 5.902 | **+12.7%** | 6.146 → 5.539 | **+11.0%** | 1.113 / 1.332 |
| `w512/skip4` | 7.119 → 6.725 | +5.9% | 6.468 → 6.401 | +1.0% | 1.421 / 1.509 |
| `w1024/skip4` | 7.322 → 6.987 | +4.8% | 6.608 → 6.638 | −0.4% | 1.496 / 1.565 |

The error is not noise and not uniform — it is **signed by lever family**.
The pooled scalar **under-rates every skip arm** (by 16.5% at `skip8`) and
**over-rates every tight-window arm** (by 11–13% at `w256`). Between those two
arms the scalar misstates their acceptance ratio by roughly **27 points** in
the quantized family.

That is precisely the bias needed to produce the failure section 15 could
only flag as a risk: on scalar acceptance the selector sees deep skip as the
worst thing in the lattice and tight windows as good, while the LO cell
measures `skip8` second-best and `w256` sixth. **The misranking is not a cost
bug and not a transfer bug — it is the selector reading acceptance at a
generation length the workload never occupies.**

### Scope

Single boot per arm; `ignore_eos` deliberately (this is calibration, not a
scored run, and it is what guarantees the long-u buckets fill). Bucket 3
spans 3–8K while LO generations reach 13–17K, so `tau_eff` still carries the
last measured bucket forward — conservative for both trends, since unwindowed
acceptance is still rising and windowed still falling at the edge. Depth 8
here against K=4 in the scored runs; the curves are used as ratios, not
levels.

---

## 17. Feeding it all back: the prediction, and four refuted explanations

Sections 14 and 16 supplied the two inputs the quantized family was missing —
cost coefficients and u-resolved acceptance. This assembles them into the
drain-integrated prediction of section 7 and scores it, now as a script
(`predict_w98_refined_lo.py`) rather than the ad-hoc analysis section 12 used.

Two corrections the script makes explicit. Acceptance is **re-derived at the
deployed depth**: the curves are measured at depth 8 while the scored runs
speculate at K=4, so a depth-8 tau of 7.9 is not a token count any scored step
could commit; it is recomputed from `pos_accepted`, which is exact rather than
approximate, since position *i* is reached only if every earlier position was
accepted. And the comparison is against the **raw** per-token ratio, not the
context-corrected score: Campaign 1's correction removes context growth from a
model-free comparison, and this model integrates context growth explicitly, so
correcting the measurement too would charge it twice. (That resolves a
discrepancy in this record: section 2's table is context-corrected, section
12's "measured" column is not. Both are right for their purpose.)

**Control first.** On the bf16 family the script returns mean absolute error
**0.0683** against section 12's hand-computed **0.0675** — the pipeline is a
faithful reimplementation, and now a reproducible one.

### The quantized family

| arm | predicted | measured | error |
| --- | --- | --- | --- |
| `w1024/skip4` | 1.437 | **1.552** | −7.4% |
| `woff/skip8` | 1.241 | **1.540** | **−19.4%** |
| `w512/skip4` | 1.346 | 1.486 | −9.4% |
| `woff/skip0` | 1.365 | 1.433 | −4.7% |
| `woff/skip4` | 1.432 | 1.425 | +0.4% |
| `w256/skip4` | 1.186 | 1.306 | −9.2% |
| `w128/skip4` | 1.194 | 1.283 | −6.9% |
| **mean abs error** | **0.0823** | | |

**The top pick is right** — `w1024/skip4`, as measured. But `skip8` is
predicted fifth where the cell measures it second, which is section 15's
misranking surviving every input this phase has added.

### Four explanations, tested and refuted

Each was plausible, each was cheap to test, and none survives.

| hypothesis | test | result |
| --- | --- | --- |
| Acceptance enters as a scalar rather than u-resolved (section 16's claim) | pool below 1K, carry flat | skip8 7th → 5th; mean error 0.0775 → 0.0823, **worse** |
| Acceptance measured on the wrong weight version | predict quantized arms with bf16 curves | mean error **0.0551**, *better*; ranking identical |
| `skip8`'s cost extrapolated past its calibration | measured the 3 missing quantized skip8 cost arms | error −18.6% → **−19.4%** |
| The u-curve under-states acceptance past 8K by carrying forward | substitute each arm's realized acceptance | error −19.4% → −15.4%, mean 0.0823 → 0.0555; **rank unchanged** |

And both inputs check out on their own terms:

* the u-curves predict the acceptance the scored runs actually realized to
  within **5%** at every one of seven arms (skip8: 3.981 predicted against
  4.200 realized, −5.2%);
* the cost fit reproduces skip8's **measured** draft chain to **1.1%**
  (22.655 ms, now measured rather than extrapolated).

So the two inputs are individually accurate to 1–5%, and their composition is
19% wrong on one arm. **The remaining error is in the model's structure, not
in its measurements**, and it concentrates on the arm with the lowest
acceptance — the one that takes the most steps, and therefore pays the
box's fixed per-step clamp the most times. That is a suggestion, not a
finding: it was not tested, and the four things that were tested all failed.

### Correcting section 16

Section 16 concluded that the misranking is "the selector reading acceptance
at a generation length the workload never occupies". That is **too strong**,
and the first row of the table above is the refutation. Moving from the
pooled scalar to the u-resolved curve does move `skip8` from last to fifth,
so the mechanism is real and signed as described — but it recovers two
places out of five, and leaves the mean error slightly worse. Acceptance was
part of the story and is not the story. The section stands as written with
this correction appended, on the same principle as section 13's: it was right
about its data and wrong about its reach.

### What would settle it

The four refutations narrow the search to the composition — the verify split,
the drain integral, or the clamp's per-step constant, none of which section 7
calibrated against anything but the parked arm. The cheapest discriminator is
a batch sweep: every one of those terms scales differently with batch, while
cost and acceptance are already pinned. That is the next measurement, and it
is also the one the refined grid was built to demand.

---

## 18. The batch sweep: a fifth refutation, and the switching case

Section 17 narrowed the residual to the composition and named the batch sweep
as the discriminator, because every uncalibrated term scales differently with
batch while cost and acceptance are now pinned. Three arms — `off`,
`woff/skip8`, `w1024/skip4` — at batches 2, 4, 8 and 16, quantized family,
Phase-100 protocol throughout. Batch stops at 16 deliberately: at 32 the run
needs ~544K KV tokens against this boot's ~355K, so the measurement would
carry preemption instead of the split.

### The split, measured rather than assumed

Section 7 solved the verify split on the parked arm at one batch, where it is
not identifiable — a single per-token number cannot say how much of a step is
paid once and how much per sequence. Batch identifies it: for the parked arm
`per_token(B) = Vs * H/T + Vp`, a straight line in the run's own
horizon-over-tokens ratio.

| | assumed (section 7) | **measured** |
| --- | --- | --- |
| shared, `Vs` | 7.69 ms | **8.79 ms** |
| per-request, `Vp` | 0.989 ms | **0.682 ms** |

The assumption was wrong in both terms — the profiler's verify divided by
batch over-attributes to per-request by 45%. **And correcting it changes
nothing**: mean error 0.0823 → 0.0813, `skip8` unmoved at −19.4%, ranking
identical. That is the fifth hypothesis tested and refuted, and the sweep was
worth running for that alone.

### What the sweep actually found

| batch | `skip8` measured | predicted | error | `w1024/skip4` measured | predicted | error |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | **0.939** | 0.961 | **+2.3%** | 1.339 | 1.292 | −3.6% |
| 4 | 1.296 | 1.140 | −12.0% | 1.516 | 1.437 | −5.2% |
| 8 | **1.540** | 1.241 | **−19.4%** | 1.552 | 1.437 | −7.4% |
| 16 | 1.402 | 1.877 | **+34.0%** | 1.607 | 1.581 | −1.6% |

**The model is not mislevelling `skip8`; it is getting its batch scaling
wrong.** Across an 8x batch range it tracks `w1024/skip4` within 8% at every
point, while its `skip8` error swings from +2.3% to −19.4% to +34.0%. A single
wrong coefficient cannot do that. The clean region is decisive on its own:
from batch 2 to 8 the arm really improves by **+64%** and the model credits it
**+29%**.

**And the measurement is non-monotone.** `skip8` peaks at batch 8 (1.540) and
falls at 16 (1.402), while every term in the cost model is monotone in batch —
shared terms fall as `1/B`, per-request terms are flat — so **a maximum is
structurally unreachable for it.** That is a shape error, not a parameter
error, and no refit can fix it.

The batch-16 point carries a caveat that must be stated: `skip8` took **6 cap
hits** there against 1 everywhere else, and emitted 308K tokens against
`w1024`'s 220K. Its low acceptance changes the sampled tokens, lengthens the
generations, and pushes requests into the 32K cap — and 16 requests averaging
~19K tokens sits close to this boot's KV ceiling. So the +34% is the least
trustworthy number in the table, and the b2–b8 trend is where the finding
rests.

### The switching case, measured

Set the model aside; the measurement is the more valuable half.

| batch | `w1024/skip4` | `woff/skip8` | winner |
| --- | --- | --- | --- |
| 2 | **1.339** | 0.939 | window, by **43%** |
| 4 | **1.516** | 1.296 | window, by 17% |
| 8 | 1.552 | **1.540** | tied (0.8%) |
| 16 | **1.607** | 1.402 | window, by 15% |

At batch 2 the deep-skip arm does not clear parity and the windowed arm wins
by 43%; by batch 8 they are inside each other's noise. **One cell, one
content distribution, one weight version, and the gap between two arms moves
by a factor of 43% on batch alone** — with the losing arm at one batch being
the co-winner at another.

This is the strongest switching evidence the phase has produced, and it is
qualitatively unlike the R-regime result where per-regime selection was worth
+1.4%. It also confirms Campaign 1's b8/b16/b32 crossover on our own lattice
rather than by analogy. What it does **not** yet show is a batch at which
`skip8` strictly wins: the flip here is from "loses badly" to "ties", so the
switching value is real but bounded by what these two arms can offer.

### Where this leaves the model

Five hypotheses tested, five refuted: scalar acceptance, wrong-family
acceptance, extrapolated skip cost, carried-forward acceptance, and now the
verify split. The inputs are each accurate to 1–5%. The failure is a shape the
model cannot express — a batch optimum — and locating it took a sweep rather
than another fit.

The honest reading for the selector: **it may be calibrated at one batch and
deployed at another only for arms whose batch response the model tracks**, and
`w1024/skip4` qualifies while `woff/skip8` does not. Since the selector's
current picks are windowed, this is a bounded risk today and a blocking one
the moment deep skip enters the candidate set — which section 15 showed it
should.

---

## 19. Fixing the batch-scaling error: a real defect repaired, the shape not

Section 18 attributed the batch-scaling error to a shape the model cannot
express. Before accepting that, two cheaper explanations were checked and
both died, and then the assumption underneath was measured rather than
reasoned about.

**The batch-16 caveat was wrong.** Section 18 flagged that point as
untrustworthy on suspicion of cap hits and KV pressure. The traces say
otherwise: **zero preemptions and zero recomputed tokens in all fourteen
runs**, and KV peaks at 12,827 blocks against the 22,190 registered — 58%.
Under the registered context-corrected currency the peak is *sharper*, not
softer:

| batch | `woff/skip8` | `w1024/skip4` |
| --- | --- | --- |
| 2 | 0.778 | 1.352 |
| 4 | 1.322 | 1.549 |
| 8 | **1.562** | 1.565 |
| 16 | 1.294 | 1.600 |

The non-monotonicity is real machine behaviour and the caveat is withdrawn.

**gamma was not it either.** The predictor had been running at `gamma = 1.0`
while section 13 measured 1.25 across three cells. Applying it moves the
batch-16 error from +34.0% to +32.8%. Sixth refutation.

### The defect, measured

Every cost arm in this phase ran at batch 8, so the draft chain's split into
batch-shared and batch-proportional parts was never identified — `split_cost`
divides a batch-8 coefficient by 8, which is an assumption, and section 18
had just caught the identical assumption in verify. The profiler reports the
draft chain per step, so sweeping batch measures it:

| batch | 2 | 4 | 8 | 16 |
| --- | --- | --- | --- | --- |
| measured `draft_chain_ms` | 18.481 | 19.182 | 22.655 | 27.382 |
| model | 20.45 | 21.10 | 22.40 | 25.00 |

**The model's batch slope is half the truth.** Refitting with batch in the
design — the KV column carrying per-step rather than per-sequence bytes —
gives a per-request coefficient **1.94x larger**, and `f_win` returns to a
physical **+2.68 ms** from the zero that non-negativity had pinned it to.
That last point matters beyond this section: section 14's negative-window
pathology was the same misattributed split seen from another angle.

### What the repair buys

| batch | arm | before | after | measured |
| --- | --- | --- | --- | --- |
| 2 | `woff/skip8` | +2.3% | **+17.9%** | 0.939 |
| 4 | `woff/skip8` | −12.0% | **−4.3%** | 1.296 |
| 8 | `woff/skip8` | −19.4% | −19.7% | 1.540 |
| 16 | `woff/skip8` | **+34.0%** | **+0.7%** | 1.402 |
| | **mean over 8 points** | **0.1069** | **0.0729** | |

**Mean error across the sweep falls by a third and the blowup is gone** —
batch 16 from +34.0% to +0.7%. The repair is a measurement, not a fit to the
target: the coefficient came from the draft chain's own profiler trace, and
the throughput predictions were not consulted in deriving it.

### What it does not buy, stated plainly

It does not fix the thing section 18 named. The predicted sequence is
1.107, 1.240, 1.237, 1.411 — monotone — against a measured 0.939, 1.296,
1.540, 1.402 that **peaks at batch 8**. A model whose every term is either
`1/B` or flat cannot produce an interior maximum at any coefficients, so this
was never a fit that could succeed. The repair moved error from batch 16 to
batch 2 and left batch 8 untouched at −19.7%.

Nor does it change the ranking. At batch 8 over all seven arms the mean error
improves 0.0823 → 0.0792 and `skip8` is still predicted fifth against a
measured second.

**So: one real defect found and repaired, and the shape error survives it.**
The remaining signature is specific enough to be worth stating for whoever
takes it next — the model is right at batch 16, 20% low at batch 8, and 18%
high at batch 2, on the arm with the lowest acceptance and the deepest KV
reads. That is a term that peaks with batch, and the natural candidate is the
one the sweep was too coarse to see: the armed verify processes `B x (K+1)`
query positions against the parked path's `B x 1`, so it meets any
compute-bound knee at a fifth of the batch. Testing that needs the verify
measured against batch on its own, which this sweep did not isolate.

---

## 20. Verify against batch: the mechanism refuted, the curvature confounded

Section 19 proposed a mechanism for the batch peak — an armed verify carries
`B * (K+1)` query positions against the parked path's `B * 1`, so it should
meet any compute-bound knee at a fifth of the batch — and said testing it
needed verify measured against batch on its own. The profiler already reports
verify per step, so the batch sweep answers it directly.

### Verify is linear. The mechanism is wrong

| batch | 2 | 4 | 8 | 16 |
| --- | --- | --- | --- | --- |
| query positions (`B x 5`) | 10 | 20 | 40 | 80 |
| `verify_ms` | 6.593 | 6.997 | 8.135 | 9.522 |

An affine fit gives `6.230 + 0.211 * B` with residuals **at or below 2.8%**
across an 8x batch range. **There is no knee.** Verify is strongly
shared-dominated — 79% of it at batch 8 is paid once — and it does not bend
between 10 and 80 query positions. Section 19's proposed mechanism is
refuted, and it was mine, so it goes in the record with the other seven.

### The step does curve — in opposite directions

What the sweep does show is in the whole step, from the traces' own decode
time:

| path | quadratic fit (ms) | curvature | affine → quadratic residual |
| --- | --- | --- | --- |
| parked, `B x 1` positions | 3.702 + 0.745B **− 0.0643B²** | **concave** | 3.72% → 1.21% |
| armed `skip8`, `B x 5` | 13.261 + 0.455B **+ 0.0834B²** | **convex** | 2.80% → **0.15%** |

The local slope of the armed step rises 0.640 → 1.996 ms per request while
the parked slope *falls* 0.571 → −0.083. **Concave over convex is exactly the
ratio that peaks**, so the shape the model cannot express is now measured
rather than hypothesised — even though the reason for it is not the one
section 19 gave.

### But the curvature is confounded with context, and does not transfer

Two attempts to put it in the model, and the second explains the first.

**Bolting the measured 0.0834 onto the batch-swept fit makes everything
worse** — mean error over the eight sweep points 7.28% → **12.25%**, batch 8
from −19.7% to −23.5%. It double-counts: a linear-in-batch fit to a convex
curve absorbs the convexity into an inflated slope, and section 19's repair
had already done exactly that.

**Fitting the quadratic jointly** — which is the correct way, and is what the
code now does — finds a curvature of only **+0.0122 ms/req²**, *6.8x smaller*
than the step-level 0.0834. It improves the middle of the range (batch 8
−19.7% → −17.2%, batch 4 −4.3% → −2.2%) and spoils the end (batch 16 +0.7% →
+10.5%), for a mean of 7.49% against the linear model's 7.28%. **And the
prediction is still monotone**: 1.113, 1.267, 1.275, 1.549.

The gap between 0.0834 and 0.0122 is the finding. The larger number comes
from LO runs where **context is not held fixed**; the smaller one from
equal-work arms where it is. And in these runs batch and realized context are
not independent:

| mean generated tokens per request | b2 | b4 | b8 | b16 |
| --- | --- | --- | --- | --- |
| `woff/skip8` | 18,584 | 19,112 | **15,061** | 19,273 |
| `w1024/skip4` | 9,279 | 19,010 | 16,317 | 13,773 |

Under natural EOS each arm generates what it generates, and `skip8`'s
generations are **shortest at exactly the batch where its speedup peaks**.
Shorter generations mean shallower contexts, cheaper KV and a better ratio.
So a large part of the "batch curvature" is context growth wearing a batch
costume, and the clean fixed-context measurement — which is 6.8x smaller —
is the one that transfers.

### Where this leaves it

Verify's batch scaling is now measured and is linear, which removes it from
suspicion permanently. The step curvature is real but mostly not a batch
effect, so no batch term will carry it. The honest statement is that **the
peak is not yet demonstrated to be a property of batch at all** — it is
confounded with what these particular requests happened to generate, and
separating them needs runs that hold generation length fixed while varying
batch, which `ignore_eos` does and the scored protocol deliberately does not.

That is a clean experiment and it is the next one. It is also a caution about
the section 18 result: the switching case there — the window arm beating deep
skip by 43% at batch 2 and tying at batch 8 — rests on measurements whose
context profiles differ by a factor of two across batches, so the *magnitude*
of that crossover should be treated as provisional even though its direction
matches Campaign 1 independently.

---

## 21. The fixed-length sweep: most of the residual was the workload

Sections 17-20 chased a 19% error on `woff/skip8` through six refuted
explanations. This section is the seventh test and the one that lands, and it
was prompted by an observation from outside the analysis: greedy sampling
does not make the arms emit the same tokens, and requests that stop earlier
change the cost being modelled.

Both halves check out. Every scored runner uses `temperature=0.0` and
`seed=0`, and **every armed arm still diverges from the parked one on 8 of 8
requests**, with output totals spanning 0.871-1.000 of OFF's. Greedy fixes
the rule for choosing a token, not the logits it chooses from; batch-composition
numerics flip near-tie argmaxes and EOS lands elsewhere. Under natural EOS
that divergence moves where each request *stops*, and generation length drives
the drain, which section 7 showed is a first-order term. So each arm was being
predicted against a different workload.

Twelve boots with `ignore_eos` and a 16,384-token budget — chosen to bracket
the natural-EOS realized mean of 15,061-19,273 — make the workload identical
by construction. Every arm at every batch emitted exactly `16384 * B` tokens,
zero cap hits.

### The contamination, measured

| batch | arm | natural EOS | **fixed length** | inflation |
| --- | --- | --- | --- | --- |
| 2 | `woff/skip8` | 0.939 | **1.159** | **−19.0%** |
| 4 | `woff/skip8` | 1.296 | 1.290 | +0.5% |
| 8 | `woff/skip8` | 1.540 | **1.277** | **+20.6%** |
| 16 | `woff/skip8` | 1.402 | 1.186 | +18.2% |
| 2 | `w1024/skip4` | 1.339 | 1.380 | −2.9% |
| 4 | `w1024/skip4` | 1.516 | 1.369 | +10.7% |
| 8 | `w1024/skip4` | 1.552 | 1.544 | +0.5% |
| 16 | `w1024/skip4` | 1.607 | **1.689** | −4.9% |

**The confound is not an offset — it swings from −19.0% to +20.6% on one
arm.** A 40-point range, larger than the effect it was contaminating. At
batch 2 `skip8` had generated 18,584 tokens per request against OFF's 9,876:
nearly twice the work at far deeper context, and the per-token ratio charged
it for that.

### Two earlier conclusions are wrong

**Section 18's switching case does not survive.** It reported the window arm
beating deep skip by 43% at batch 2 and *tying* at batch 8, and called that
the phase's strongest switching evidence. At equal work there is no
crossover: `w1024/skip4` leads at **every** batch — by 19.1%, 6.1%, 20.9% and
42.4% — and its lead is widest where section 18 said the arms tied. The
"crossover" was the workload difference. Campaign 1's batch crossover is
independent evidence and stands; ours does not.

**Section 18's shape error is real but four times smaller.** At equal work
`skip8` still peaks in the interior — 1.159, **1.290**, 1.277, 1.186, peaking
at batch 4 and falling 8.1% by batch 16 — so a model monotone in batch still
cannot express it. But the amplitude is an 11% rise and an 8% fall, not the
64% rise section 18 reported. Section 20's verdict that the peak might not be
a batch property at all is now settled: it is one, and it is minor.

### The model was never as wrong as it looked

Predicting the fixed-length runs, everything else unchanged:

| batch | `skip8` predicted | measured | error | `w1024` error |
| --- | --- | --- | --- | --- |
| 2 | 1.347 | 1.159 | **+16.2%** | −2.3% |
| 4 | 1.295 | 1.290 | **+0.4%** | +1.2% |
| 8 | 1.232 | 1.277 | **−3.5%** | −4.9% |
| 16 | 1.145 | 1.186 | −3.4% | −6.8% |
| | **mean** | | **4.84%** | (3.37% excluding batch 2) |

**Mean error falls from 7.28% to 4.84%, and the error that drove sections
17-20 — `skip8` at batch 8 — collapses from −19.7% to −3.5%.** Six
hypotheses were refuted because the thing being explained was mostly not
model error. The model's inputs were sound, its structure was adequate, and
the residual was the workload moving underneath it.

What remains is small and specific: `skip8` at batch 2 is over-predicted by
16.2%, and the model has `skip8` monotone *decreasing* (1.347, 1.295, 1.232,
1.145) where measurement rises then falls. It gets the fall right and the
rise wrong.

### The methodological point, which is the durable part

The phase already knew to calibrate with equal work and score with natural
EOS. What it had not done was apply that discipline to an *analysis*: sections
18-20 compared arms across batches using scored runs, where the workload is a
free variable. Any comparison that holds the machine fixed and lets the
workload move is measuring both.

A consequence for the selector, which is new and should be registered: **the
generation-length distribution is a required input, not a cell label.** It
drives the drain, it differs per arm through numerics the selector does not
control, and on AIME it spans 4,721-32,768 tokens — a 7x range within one
cell. A selector deployed on a different long-output workload needs that
distribution. The good news measured in the same pass: predictions fed the
*parked* arm's lengths — which a selector can actually obtain — score
**0.0435** against **0.0792** for each arm's own realized lengths, so the
input a deployment can supply is the better one.

---

## 22. Is acceptance batch-dependent? Yes, and it is numerical

Two objections to section 21, both from outside the analysis, and they point
at the same gap.

**First: `ignore_eos` should inflate acceptance**, because past the natural
stopping point the continuation is degenerate and a draft finds it easy. The
opening-of-run comparison says the two protocols are not merely close but
*identical* — 0.6570/0.6570, 0.6655/0.6655, 0.8230/0.8230, 0.8464/0.8464 at
batches 2/4/8/16 — which is expected, since the runs are the same until the
first request would have stopped. In the tail the fixed-length runs measure
*lower* acceptance than natural-EOS ones (`skip8` 0.781 against 0.897 in the
last decile), not higher. But that comparison is not controlled: a
natural-EOS tail contains only the surviving long requests, and survivorship
is its own selection toward predictable content. **So the inflation is not
visible here, and this data cannot rule it out** — separating them needs
per-request acceptance, which the trace does not carry.

**Second, and the real question: why does acceptance differ by batch at all?**
It does, at fixed `u`, on the same prompts, under either stopping rule:

| arm | b2 | b4 | b8 | b16 | b16/b2 |
| --- | --- | --- | --- | --- | --- |
| `woff/skip8` | 0.6570 | 0.6655 | 0.6746 | 0.6934 | **1.055** |
| `w1024/skip4` | 0.8230 | 0.8227 | 0.8385 | 0.8464 | **1.028** |

In this design batch *is* content — batch B means the first B prompts — so
that table is equally consistent with prompts 3–16 simply being easier to
draft than prompts 1–2. **Replicating one prompt to fill the batch separates
them**, holding content exactly fixed while batch varies:

| arm | b2 | b4 | b8 | b16 | b16/b2 | confounded |
| --- | --- | --- | --- | --- | --- | --- |
| `woff/skip8` | 0.7220 | 0.7742 | 0.7594 | 0.7646 | **1.059** | 1.055 |
| `w1024/skip4` | 0.7825 | 0.7684 | 0.8268 | 0.8043 | **1.028** | 1.028 |

**The effect survives intact — 1.059 against 1.055, 1.028 against 1.028.** It
is not content. It is the machine.

The mechanism is batch non-invariance, and speculative decoding is unusually
exposed to it. Acceptance is the rate at which two argmaxes agree, and the two
paths run at *different shapes by construction*: the draft proposes at
`(B, 1)` query positions while the target verifies at `(B, K+1)`. Reduction
order in the GEMM and attention kernels depends on those shapes, so the two
paths' rounding diverges differently as batch grows, and near-tie argmaxes
agree more or less often. That also explains why the effect is **not
monotone** — 0.7220, 0.7742, 0.7594, 0.7646 — since kernel tiling changes in
steps, not smoothly. It is a step function of batch shape wearing the costume
of a trend, and reading a smooth curve through it would be a mistake.

### What this means for runtime switching

This is the practical point, and it is a gap in the design rather than in the
measurement. The selector consumes `tau(u)`. The evidence here is for
**`tau(u, B)`**: acceptance moves with concurrent batch, and it moves by
*different amounts per lever* — 5.9% for deep skip against 2.8% for the
window arm over an 8x batch range.

Under natural EOS the batch drains continuously — LO runs go from 8 active to
1 — so **every long-running request crosses a range of batches, and the
acceptance ranking between levers shifts underneath it while it runs.** That
is a switching trigger that owes nothing to the workload changing: the same
request, the same content, a different optimum because its neighbours
finished. Nothing in the current cost model can see it, since `tau` enters as
a function of `u` alone.

The honest sizing: the lever *differential* is 5.9% − 2.8% ≈ **3 points of
acceptance over an 8x batch swing**, which is real but modest — comparable to
the +1.4% that per-regime switching was worth, and far smaller than the cost
side's batch dependence, where shared terms amortise over the drain. So
`tau(u, B)` belongs in the model as a correctness matter, and the switching
value it unlocks on its own is likely small. The larger switching case remains
the cost side, which section 21 showed is also where the measurement discipline
has to be tightest.

---

## 23. Correcting section 15: the risk is in the ranking, not the elimination

No new measurement. This is a re-reading of section 15's mechanism against
the code that implements the two rounds, and it moves a finding from one
stage of the selector to another without changing its size.

### What section 15 claimed

> Round 1 eliminates candidates by predicted cost, from coefficients fitted
> on one weight version. [...] **The elimination round can discard the
> runner-up of the family the selector actually deploys.**

That names the wrong stage, and the arithmetic says so directly.

### Round 1 cannot eliminate `skip8`, in either family

The registered rule is `(K+1)/q_lo < 1 + epsilon_arm`
(`w98_cost_model.eliminate`, `epsilon_arm = 0.015`): a candidate dies only
when its most favourable possible speedup — full acceptance at the
optimistic end of its predicted cost — still fails to beat parity. The rule
is one-sided by construction, and it fires on cost being too **high**.

Deep skip moves cost the other way. Section 15's own trace table:

| | draft chain, `skip0` | draft chain, `skip8` |
| --- | --- | --- |
| `target-matching` | 36.844 ms | **29.135 ms** (−20.9%) |
| `w4a16-quantized` | 26.659 ms | **22.655 ms** (−15.0%, measured in §17) |

`skip8` is the *cheapest* arm in both families. No cost-based elimination
rule can fire on the cheapest arm in the lattice, and this one is not close
to firing. Round 1 passes `skip8` through in both families, and always would
have.

The bf16 arm's sub-parity score (0.914) is an **acceptance** collapse —
0.992 → 0.762, −23.2%, bought for a 20.9% cost saving. Round 1 never sees
acceptance. That is the asymmetry the whole design rests on, and it is
working as specified here.

### Where the risk actually sits

The selector's pick is the argmax of the **predicted value map**
(`score_w98_g98e_d3.py:100`, `selector_choice`) — the cost model composed
with measured acceptance and integrated over the drain. A misranked arm is
never eliminated; it simply never wins, and therefore never reaches
confirmation. Section 17 measured exactly that: the quantized prediction
ranks `skip8` **fifth** where the cell measures it **second**, at mean
absolute error 0.0823.

So the failure is in the **join** of the two rounds, not inside either one.

### What is unchanged

Everything section 15 measured. The lattice does reorder across weight
versions (Spearman **+0.286**, against **+0.902** for a change of workload),
`skip8` does move from worst to runner-up, and section 14's finding that the
cost model is refuted across the quant axis stands untouched. The defect is
the same size; only its address is corrected.

### What it changes about priority

It separates two things that section 15 fused, and they need opposite
amounts of work:

* the **sound elimination rule** needs none. It cannot be made unsound by a
  reordering lattice, because a reordering that makes an arm *better* is
  invisible to a rule that only kills arms too expensive to pay;
* the **ranking** carries the entire error, and it is fed by a cost model
  fitted on one weight version and refuted across the other. That is where
  the quant-axis refit is worth spending on.

Section 15 stands as written with this correction appended, on the same
principle as sections 13 and 16: it was right about its data and wrong about
its reach.

---

## 24. Step 0a: the per-family fit, and what the equal-work design cannot say

Section 14 fitted both weight versions onto one surface, with the family
entering through a single byte column `W_layer * kappa_w`, and refuted it:
`f_win` went negative. The implied fix is to stop sharing a surface — Round 1
never needed one, it needs composed cells predicted from singles — so this
fits each family on its own points. No GPU: `g98_equalwork` (10 arms, bf16)
and `g98_equalwork_q4` (9 arms) were already measured.

The fix does not work, and why it does not is more useful than the fix would
have been.

### The per-family fit, in the registered form

| | `A` | `f_win` | `kappa_kv` | `F` | mean abs residual | physical |
| --- | --- | --- | --- | --- | --- | --- |
| `target-matching` | 26.36 ms | **+1.51 ms** | **+1.035e−11** | 3.79 ms | **0.345%** | yes |
| `w4a16-quantized` | 20.72 ms | **−4.56 ms** | **−2.96e−12** | 7.80 ms | 0.523% | **no** |

**Correcting section 14's attribution.** The negative `f_win` was reported as
what happens when the fit is forced across weight versions. It is not: it is
present **within the quantized family alone**, on nine of its own arms. Under
a non-negativity constraint the quantized fit is physical and 2.6x worse
(`f_win` pinned to 0, residual 0.523% -> 1.338%). Fitting per family, which
is what section 14's refutation implies, does not remove the defect.

### The cause: the window term is not a bytes integral

The quantized family's own equal-work draft chains, at `keep = 1`:

| window | `target-matching` | `w4a16-quantized` |
| --- | --- | --- |
| off | 36.844 ms | 26.659 ms |
| 1024 | 33.073 ms | **23.513 ms** |
| 256 | **32.033 ms** | 24.026 ms |

bf16 is ordered by bytes — the tighter window is cheaper. The quantized
family inverts: the **wider** window measures cheaper. No model with a
positive per-byte coefficient can produce that, so the fit produces a
negative one.

Refitting with a free per-window offset — `D = keep*(A + g_w) + F`, `g_off`
fixed at 0, which assumes nothing about bytes — separates what is resolved
from what is not:

| | `g256` | `g512` | `g1024` | mad | cond |
| --- | --- | --- | --- | --- | --- |
| `target-matching` | **−4.544** ± 0.151 | −4.296 ± 0.215 | **−3.454** ± 0.151 | 0.315% | 21.8 |
| `w4a16-quantized` | −2.823 ± 0.165 | — | −3.138 ± 0.165 | 0.523% | 20.8 |

* bf16 is ordered by bytes and the `512 -> 1024` step is **resolved**
  (−0.842 ± 0.215); `256 -> 512` is not (−0.248 ± 0.215).
* the quantized inversion is **NOT resolved**: `g256 − g1024 = 0.315 ±
  0.165`, 1.9 sigma. The quantized family's window-*size* axis is flat within
  the resolution this design has.

**So this is an identifiability finding, not a physics one.** Forced onto an
axis carrying no resolved signal, the bytes form has nothing to fit but
noise, and it reports that noise as a negative `kappa_kv`. The corroborating
number is the same in both families: the `f_win`/`kappa_kv` correlation is
**−0.956**, because with three window levels that both saturate against an
8K context the indicator column and the bytes column are nearly the same
column. Only their *net* per-window effect is identified — in either family.

### One thing the offsets do resolve, and it matters

The same window over the same context moves the same KV bytes in both
families — shared target KV, identical attention work. If the window's saving
were a KV-bytes effect it would be the same number of milliseconds in both.
It is not:

```text
g256:   -4.544 (bf16)  vs  -2.823 (w4a16)   difference 1.721 +/- 0.224  (7.7 sigma)
```

**Windowing saves 38% less in the quantized draft, on identical KV traffic.**
That is section 14's interaction, arriving from an independent direction and
now resolved rather than inferred — and it is a second, cleaner refutation of
the pure-bytes account that section 14 argued against on bandwidth grounds.

### `F` contradicts a registered claim, and this design cannot adjudicate it

The registered model has `F` quant-independent **by construction**: `lm_head`
and embeddings stay bf16 in both drafts. Measured per family:

```text
F(bf16)  = 3.801 +/- 0.544 ms
F(w4a16) = 7.797 +/- 0.592 ms      difference 3.996 +/- 0.804  (5.0 sigma)
```

Taken at face value that refutes the construction. But the obvious
alternative is that `F` is absorbing curvature in `keep`, and testing it
dissolves every estimate:

| | `A` | `keep^2` | `F` |
| --- | --- | --- | --- |
| bf16, linear | 32.835 ± 0.615 | — | 3.801 ± 0.544 |
| bf16, `+keep^2` | 27.162 ± **18.654** | 3.191 ± **10.486** | 6.296 ± **8.221** |
| w4a16, linear | 18.862 ± 0.669 | — | 7.797 ± 0.592 |
| w4a16, `+keep^2` | −6.950 ± **15.959** | 14.519 ± 8.971 | 19.150 ± **7.033** |

Nothing survives. The preregistration predicted exactly this in advance —
"an intercept cannot be separated from a slope over that span at the measured
noise" — which is why **skip16** was added to the Round-2 fit set to extend
the keep range to 0.556. The equal-work sweep has keeps {1.000, 0.889,
0.778}, a 22% span, and **no skip16 in either family**.

So `F`'s family difference is measured but not attributable. The cheapest
thing that would settle it is `woff/skip16` in both families: **two boots.**

### What Step 0a delivers

A per-family cost fit that reproduces each family's own draft chain to
**0.315%** and **0.523%** — inside this box's ~1% armed-arm stability — on a
form that claims nothing the data does not support. That is a usable Round-1
cost input for the quantized family, which the phase did not previously have.

What it does not deliver is a **bytes** model of the window, in either
family. The price of the honest form is that it predicts only at windows that
were measured, per family. Given that the response is not monotone in bytes,
that price is not a limitation of the form — it is the finding.

---

## 25. Step 1: at equal work the lattice does not reorder at all

Section 15 reported Spearman **+0.286** between the two draft weight
versions' rankings of the same seven configurations — against **+0.902** for
a change of task — and concluded that the weight version reorders the lever
lattice more than the workload does. Section 17 then found the prediction
ranking `skip8` fifth where the cell measured it second, and sections 18-20
spent five refuted hypotheses on that error.

All of it was measured under **natural EOS**, where each arm emits a
different number of tokens: 104,530 to 159,170 against the parked arm's
134,139. Section 21 established that a comparison holding the machine fixed
while the workload moves is measuring both, and retired two conclusions
drawn that way — but it only had two arms. This re-takes all seven, in both
families, at equal work.

Sixteen boots, `ignore_eos`, 16,384 tokens per request, batch 8, same
prompts, same cell.

### The gates first

| | |
| --- | --- |
| tokens emitted, every arm, both families | **131,072** (identical) |
| cap hits | **0** |
| context-ratio deviation from the parked arm | **0.000000** |
| the two families' own parked boots | 1.606 vs 1.594 ms/token (**0.753%**) |

With identical generation lengths the Campaign-1 context correction becomes
a common factor, so raw and corrected rankings coincide and `score_vs_off`
can be read directly as a ranking.

### The result

| arm | bf16 | w4a16 |
| --- | --- | --- |
| `w1024/skip4` | **1.3225** | **1.5678** |
| `w512/skip4` | 1.2890 | 1.4701 |
| `w256/skip4` | 1.2551 | 1.4082 |
| `w128/skip4` | 1.1776 | 1.3689 |
| `woff/skip4` | 1.1619 | 1.3581 |
| `woff/skip0` | 1.1415 | 1.3526 |
| `woff/skip8` | 1.0312 | 1.2796 |

**Spearman +1.000 (p = 0), against +0.286 under natural EOS.** The two
families rank the seven arms **identically** — and the order is clean:
monotone in window size, then skip4 over skip0 over skip8.

**Section 15's central claim is refuted.** Changing the draft's weight
version does not reorder the lever lattice. The reorder was the workload.

### `skip8` does not flip

Section 15's most striking number was `woff/skip8` moving from the only
sub-parity arm in the bf16 family (0.914) to the second-best quantized arm
(1.562). At equal work it is **last in both families** and above parity in
both (1.0312, 1.2796). Nothing flips.

The reason is visible in the contamination, which is signed by arm *and* by
family and spans 33 points:

| arm | bf16 natural | bf16 equal | inflation | w4a16 natural | w4a16 equal | inflation |
| --- | --- | --- | --- | --- | --- | --- |
| `w1024/skip4` | 1.4960 | 1.3225 | +13.12% | 1.5651 | 1.5678 | −0.17% |
| `w512/skip4` | 1.4213 | 1.2890 | +10.26% | 1.5087 | 1.4701 | +2.63% |
| `w256/skip4` | 1.1127 | 1.2551 | −11.35% | 1.3322 | 1.4082 | −5.40% |
| `w128/skip4` | 1.0506 | 1.1776 | −10.78% | 1.3011 | 1.3689 | −4.95% |
| `woff/skip4` | 1.0424 | 1.1619 | −10.28% | 1.4426 | 1.3581 | +6.22% |
| `woff/skip0` | 1.0524 | 1.1415 | −7.81% | 1.4486 | 1.3526 | +7.10% |
| `woff/skip8` | 0.9141 | 1.0312 | −11.36% | **1.5617** | 1.2796 | **+22.05%** |

The largest single contamination in the table, **+22.05%**, sits on exactly
the arm whose behaviour drove section 15's flip, section 17's misranking and
five refuted hypotheses in sections 18-20.

**A second finding, which is methodological and transfers.** Every number in
those columns is already **context-corrected** under Campaign 1's amendment
1 (`CTX_SHARE = 0.15`). The correction is not enough at this cell's
magnitudes: with arms emitting 104K to 159K tokens, a 15% context share
leaves up to 22 points of length confound standing. **At LO lengths the
registered correction does not substitute for equal work.**

### The prediction, re-scored

The same model, the same coefficients, the same u-curves — only the workload
held fixed:

| | mean abs error, natural EOS | **equal work** |
| --- | --- | --- |
| `target-matching` | 0.0683 | **0.0218** |
| `w4a16-quantized` | 0.0823 | **0.0366** |

and `woff/skip8` in the quantized family, the residual that drove sections
17-20, goes from **−19.4%** to **+5.6%**.

Ranking, per family:

* **bf16 is right** up to one adjacent pair — `w128/skip4` and `woff/skip4`
  are predicted 1.169 against 1.173, a 0.3% gap this box cannot resolve;
* **the quantized family is still wrong**, but no longer diffusely. The
  model over-predicts the two **unwindowed** arms (+7.6%, +4.5%) while
  getting the windowed ones to within 2% (`w256` +0.4%, `w128` −1.9%), so it
  ranks `woff/skip4` and `woff/skip0` above `w256` and `w128` where
  measurement puts them below.

That residual lands exactly on the term section 24 independently showed is
unphysical in that family: the quantized window offsets, fitted where the
window axis had saturated, with `f_win` and `kappa_kv` both negative. Two
sections reached the same defect from opposite directions.

### What this does to the plan

Step 1 was registered with a branch: Spearman toward +0.9 means section 15's
crack is a correctness matter rather than a live selector risk; near +0.3
means the quant-axis model is the phase's main open defect. **It returned
+1.000**, so the branch resolves to the first reading, and Step 2 shrinks
accordingly.

What survives, and should not be over-corrected away:

* **section 24's cost finding stands** — the same window over identical KV
  traffic saves 4.544 ms in the bf16 draft and 2.823 in the quantized one —
  and it now has a sharper consequence than section 15 gave it: it is the
  *remaining* ranking error in the quantized family, worth two rank places,
  not a reordering of the whole lattice;
* **the top pick is `w1024/skip4` in both families under both protocols.**
  That is why the 31-cell selector scored 0.982 without ever having been
  checked across weight versions;
* **natural EOS is still what a deployment sees.** These numbers do not say
  the natural-EOS measurements were wrong; they say the *cause* is the
  generation-length distribution, not the weight version. The selector must
  still cope with it — as section 21 registered, by taking the length
  distribution as an input — but it must not model it as a quant effect.

**Scope.** One cell, one batch, single boots. Equal work is the calibration
protocol; the scored protocol keeps natural EOS. The claim here is about the
lattice's structure and the model, not about deployed throughput.

---

## 26. The kernel trace: identical KV, identical attention, unrealized savings

Section 24 measured the window saving 4.544 ms in the bf16 draft against
2.823 in the quantized one and called it an exposure effect, reading the
bandwidth arithmetic as "the KV read is 81% exposed in one family and 50% in
the other". The objection to that is immediate and correct: **the KV cache is
shared, so at the same window over the same context the KV traffic is
identical by construction.** Nothing about quantizing the draft's weights
touches it.

This settles it at kernel granularity. Four arms — `{bf16, w4a16} x {woff,
w256}` at keep 1 — on the equal-work sweep's content and batch, warmed to
context 4158-4160 so the window does the job whose cost section 24 fitted,
then 25 steps under `torch.profiler`.

### The attention kernel costs the same in both families

| arm | attention kernel | calls |
| --- | --- | --- |
| `bf16 / woff` | **9.764 ms/step** | 180 |
| `q4 / woff` | **9.912 ms/step** | 180 |
| `bf16 / w256` | 4.029 ms/step | 180 |
| `q4 / w256` | 3.817 ms/step | 180 |

180 calls is 36 layers x (4 draft forwards + 1 verify), which confirms the
chain structure the bandwidth account assumed. Unwindowed, the two families'
attention differs by **1.5%**. The window removes **5.735 ms** (bf16) and
**6.095 ms** (w4a16) — 6% apart, and both within 9% of the **5.607 ms** that
18.78 GB of removed KV takes at this box's 3.35 TB/s peak.

**So the bytes account is right, and it is right in both families.** The KV
read is fully exposed and fully explained by its traffic. Section 24's
"partly exposed" reading was wrong.

Two controls confirm the arms are otherwise identical: GEMM time is
untouched by windowing (bf16 27.525 -> 27.561 ms, w4a16 18.106 -> 18.118),
and the kernels windowing *adds* — gather, scatter, elementwise for the
window compaction — total **0.13 ms/step** and are the same in both
families.

### Where the difference actually is

The draft chain holds 144 of the 180 attention calls, so apportioning:

| | attention removed | draft-chain share | **fitted `g256`** | unrealized |
| --- | --- | --- | --- | --- |
| bf16 | 5.735 ms | 4.588 ms | **4.544 ms** | **+0.044** |
| w4a16 | 6.095 ms | 4.876 ms | **2.823 ms** | **+2.053** |

**In the bf16 draft the window's measured saving IS the attention time it
removes, to 1.0%.** In the quantized draft, **42% of the freed GPU time never
appears in the chain.**

That is the whole anomaly, and it is not about KV at all. It is about
whether freed device time converts into wall time. Total kernel time per
step:

```text
bf16   woff 41.950 -> w256 36.335 ms
w4a16  woff 31.129 -> w256 25.160 ms
```

The quantized chain runs **26% less device work**, having already shed 41
GB/step of weight traffic. A step with less GPU work has less behind which to
hide its host work — the per-step window compaction, the metadata rebuild,
the launch stream. Freeing 4.9 ms of GPU time in a chain already close to
host-bound converts only partly into wall time. The idle deltas point the
same way (+14.48 ms bf16, +17.66 ms w4a16 when windowing is on), though the
profiler inflates host time far too much for those magnitudes to be used
directly.

The self-consistency is worth noting: the bf16 fit needs the draft chain to
hold 79.2% of the attention saving, and 144/180 is 80.0%. The apportionment
was not tuned to produce that.

### Correcting section 24, and what it means for the model

Section 24 stands on its measurements — the fitted offsets, their standard
errors, the 7.7 sigma family difference — and its *mechanism* is wrong. The
KV read is not partly hidden. It is fully exposed in both families, and the
window's value differs because the **freed time is realized differently**.

The structural consequence is sharper than section 24's version:

* the fitted per-window offset is **not a memory-traffic quantity**. It is
  (attention time removed) − (host time exposed), and only the first term is
  family-independent and predictable from bytes;
* that is why the quantized window axis **saturated** in section 24 — once a
  chain is host-bound, tightening the window frees GPU time that does not
  convert, so `w256` and `w1024` measure the same;
* and it is why section 25's residual ranking error sits exactly on the
  `woff`-versus-windowed comparison in the quantized family, and nowhere
  else.

**The window term is therefore not separable per lever.** How much a window
is worth depends on how much device work the rest of the step has, which the
other two levers set. A cost model that fits `f_win` once per family, as
every version in this phase has, is fitting a quantity that moves with skip
depth and weight version together.

### Scope

One boot per arm, 25 profiled steps, one context. `torch.profiler` with CPU
activities inflates host time heavily — wall is 88-97 ms/step here against a
real chain-plus-verify of roughly 30-37 — so only the **device** numbers and
the cross-arm ordering are used above; no wall or idle magnitude is claimed.
The 144/180 apportionment assumes the draft's `q=1` and verify's `q=5`
attention calls cost about the same per call, which is expected for a
KV-read-bound kernel and is corroborated by the bf16 reconciliation landing
at 79.2% against the assumed 80.0%.

---

## 27. The step budget: what quantization buys, and what it cannot

The same four traces, bucketed by component. The question is the obvious one
to ask of section 26's result: if quantizing the draft removes weight traffic,
why does the attention kernel not get faster too?

### The budget, unwindowed

| component | bf16 | w4a16 | delta |
| --- | --- | --- | --- |
| **draft body GEMMs** | 20.253 | 10.827 | **−9.426** |
| attention | 11.005 | 11.092 | +0.087 |
| verify GEMMs | 5.606 | 5.601 | −0.005 |
| `lm_head` | 1.666 | 1.678 | +0.011 |
| other | 3.421 | 1.932 | −1.489 |
| **total device** | **41.950** | **31.129** | **−10.821** |

Quantization moves **one bucket**. Everything the draft's weights do not
touch is unchanged inside noise: attention +0.8%, verify GEMMs −0.09%
(the *target's* weights stay bf16 in both), `lm_head` +0.7%.

**The `lm_head` row settles a question section 24 left open.** The registered
model declares `F` quant-independent *by construction*, and section 24
measured a fitted `F` of 3.801 ms against 7.797 — a 5 sigma difference that
would have refuted it, except that adding `keep^2` dissolved every estimate.
The kernel says the construction is right: the `lm_head` GEMM is **1.666 vs
1.678 ms**, four calls per step at ~417 us each in both families. So section
24's fitted difference was an artifact of a 22% keep span, and the two
`woff/skip16` boots proposed there are **no longer needed to establish
quant-independence** — only to pin `F`'s magnitude, which no claim currently
rests on.

### Why attention does not get faster

Because kernels are **serialized**. Within a CUDA stream the attention kernel
and the GEMM kernels run one after another, never together, so attention
already had the entire HBM bandwidth to itself. There was no contention for
quantization to relieve. Quantization shortens the GEMM kernels; it cannot
shorten a kernel whose time is KV bytes over bandwidth when neither term
moved. Measured, attention is 0.8% *slower* under quantization — the opposite
sign from contention relief, and inside noise.

Stated generally, and worth keeping: **each lever shortens only the kernels it
touches.** Quantization -> draft body GEMMs. Window -> attention. Skip ->
every per-layer kernel, proportionally. The levers do not share bandwidth
headroom, because they do not run at the same time.

### Quantization delivers 30% of the speedup its bytes promise

| | draft weight traffic | time | achieved bandwidth |
| --- | --- | --- | --- |
| bf16 (nvjet) | 55.57 GB/step | 20.253 ms | 2,744 GB/s — **81.9% of peak** |
| w4a16 (Marlin) | 14.32 GB/step | 10.827 ms | 1,323 GB/s — **39.5% of peak** |

**3.88x fewer bytes, 1.87x faster.** Marlin sustains less than half the
bandwidth the bf16 GEMM does: it dequantizes on the fly, so it spends far
more compute per byte, and at batch 8 — query width 1 per request, so M = 8 —
there is nothing to amortize that against.

This is the direct, measured reason section 14's cross-family byte column
could not work. A single `kappa_w` assumes both families read weights at the
same rate. They read at **82%** and **40%** of peak, so a coefficient fitted
across them attributes a *rate* difference to a *byte* difference. Section
14's bandwidth argument inferred this from an impossible implied figure of
4050 GB/s; the trace measures it directly.

### The synthesis

Section 26 found that quantization makes the window worth 2 ms less without
touching the KV traffic. This says how: quantization removes 10.8 ms of
**device** work while leaving attention alone, and thereby shrinks the pool of
device work available to hide **host** work behind.

So the levers do interact, and the channel is now named: **they interact
through the host, not through the GPU.** On the device their effects are
cleanly separable, which is why per-kernel accounting adds up. The
non-separability appears only when device time is converted to wall time —
which is the conversion the cost model performs and never measured.

### The consequence, predicted and then tested

If host exposure is what makes a window worth less, a *deeper skip* should do
to a window what quantization does, for the same reason: fewer layers, less
device work, less to hide behind. The model scales the window offset
proportionally with `keep`, so the prediction is that the saving **per unit
keep** should fall as keep falls. The equal-work sweep already has three
windows at three keeps in both families, so this cost no boots.

Window saving per unit keep, `(D_off - D_w) / keep`, in ms:

| family | window | keep 1.000 | keep 0.889 | keep 0.778 | trend |
| --- | --- | --- | --- | --- | --- |
| bf16 | w256 | 4.811 | 4.540 | 4.105 | **-14.7%** |
| bf16 | w1024 | 3.771 | 3.359 | 3.052 | **-19.1%** |
| w4a16 | w256 | 2.633 | 2.727 | 3.261 | +23.8% |
| w4a16 | w1024 | 3.146 | 3.086 | 3.190 | +1.4% |

**Confirmed in bf16, refuted in w4a16.** The bf16 family falls
super-proportionally at both windows, in agreement, by 15-19% over a 22% keep
range -- the model's proportional scaling of `f_win` is wrong in the direction
predicted. The quantized family does not fall at all.

The reading that fits both is section 26's, applied twice: the quantized chain
is **already** host-bound at `keep = 1` -- that is what "42% of the freed GPU
time never appears" means -- so removing further device work cannot expose more
host time, because there is none left hidden. bf16 still has device work to
lose, so skipping layers walks it toward the same floor and the window loses
value on the way.

That is a post-hoc reading and is marked as one. What is not post-hoc: the
bf16 trend was predicted before it was computed, it appears at both windows
with the same sign, and it is 3-4x the 0.32 ms noise on a difference of two
arms. The `w4a16` w256 rise is ~2 sigma on the same footing and its w1024
counterpart is flat, so the quantized family's non-result is genuinely a
non-result rather than a counter-trend.

**For the model this is concrete:** `f_win` is not a per-family constant. In
the bf16 family it needs a `keep` interaction; in the quantized family it does
not, because that family sits against a floor. A single fitted `f_win` per
family, which is what every version in this phase uses, is an average over a
range it varies across by 19%.

---

## 28. Transferability: what moves to another box, model, or workload

Written before starting step 3, because it is the first thing a reader asks
and because sections 25-27 supply the strongest evidence the phase has for
answering it. Nothing here is new measurement.

| axis | transfers | cost to move |
| --- | --- | --- |
| the two-round **method** | **yes**, structurally | nothing |
| cost **coefficients** | **no** | one singles sweep, ~19 boots |
| **acceptance** curves | across hardware yes, across content no | re-measure per workload |
| serving **stack** | weakest link | refit, possibly restructure |
| **MoE**, TP>1 | **unvalidated** | structural extension |

### The method transfers because the asymmetry is structural

Cost is set by bytes and kernels; acceptance by content. That is not a
property of this box. And the elimination rule `(K+1)/q_lo < 1 + epsilon_arm`
kills only what cannot pay at *perfect* acceptance, so it is sound wherever
the cost interval is honest — it is arithmetic, not calibration. Section 25
is the positive evidence: at equal work the model predicts to 0.0218 and
0.0366 and ranks the lattice identically in both weight versions.

### The coefficients do not, and sections 26-27 say exactly why

Three independent measurements each defeat coefficient transfer:

* **`kappa_w` is a rate, not a byte constant.** Marlin sustains **39.5%** of
  peak against the bf16 GEMM's **81.9%** — a property of (kernel, GPU
  generation, batch shape). It is why 3.88x fewer bytes buys 1.87x less time.
* **The window term carries a host-exposure component** — 42% of its value
  in the quantized family — which depends on host CPU, driver and runtime
  mode. This box has a documented clamp (a fixed per-step host delay, the
  9-16% OFF-versus-armed band) that h103/h104 do not.
* **`f_win` is not constant even within one family**, varying 19% across the
  keep range in bf16.

**But the design was built to be re-fitted.** Round 1 consumes single-lever
profiles by construction, so relocating costs one singles sweep: ~19 cost
boots, ~14 acceptance boots, 4 for the verify split — **about half a day** at
this box's measured boot times — and it buys predictions for 30 composed
cells.

### Per axis

**Server.** Acceptance transfers exactly; G98-F measured bit-identical accept
patterns across boxes. Every cost coefficient does not. One caveat of our
own: section 22 showed acceptance moves up to 5.9% from batch non-invariance
alone, so a different GPU's kernel tiling can shift it at that scale.

**Model.** Two of the decisive quantities are **computable before booting
anything**:

```text
KV bytes/token = 2 * 2 * n_kv * d_head * L = 147,456   (Qwen3-8B)
lm_head bytes  = V * H * 2 = 151936 * 4096 * 2 = 1.245 GB
```

The second predicted its own measurement: 1.245 GB read four times per step
is 1.49 ms at peak, and `lm_head` measures **1.666 ms** — 89% of
bandwidth-bound. So a new model can be screened in advance: MQA has far less
KV traffic and a smaller window ceiling; a smaller vocabulary has a smaller
irreducible floor. What still needs measuring is skip tolerance — a
redundancy property of the weights — and the knapsack's additivity boundary,
which for this model is exact at k=4 and k=8 and inverts at k=16.

The sharpest illustration of why byte-share reasoning alone fails is in this
model's own numbers. At batch 8 and context 4252, KV is **25%** of traffic in
the bf16 draft and **57%** in the quantized one. By traffic share the window
should matter *more* under quantization. It delivers *less*.

**Workload.** This is what the selector is, so the design handles it by
construction — subject to section 21's registered input requirement, the
generation-length distribution, taken from the parked arm (0.0435 against
0.0792 for each arm's own). Section 25's protocol finding travels with it: at
LO lengths the registered context correction leaves up to 22 points of
confound standing, so equal work is not optional for calibration.

**Serving stack.** The least transferable and the real risk. Host exposure is
a property of the stack's per-step host work; on another stack it is a
different quantity, not a different number.

### What is genuinely unvalidated

One model (Qwen3-8B dense), one quantization kernel (w4a16 Marlin), one stack
(vLLM V1, piecewise, shared KV), TP=1, and two boxes whose absolute numbers
are explicitly not comparable. **MoE is a structural extension, not a
re-fit**: under routing, weight traffic per token is not a constant, so
`kappa_w` is not a coefficient at all.

### What does port, because it is mechanistic

1. **Kernels serialize**, so each lever shortens only the kernels it touches.
2. **Dequantizing GEMMs sustain lower bandwidth than bf16** — universal in
   kind, varying in fraction.
3. **Removing device work pushes a step toward host-bound**, so levers
   interact through the host rather than through the GPU.
4. **Acceptance is batch non-invariant**, inherent to speculative decoding
   where the draft runs at `(B, 1)` and verify at `(B, K+1)`.
5. **`F` is a floor no lever touches**, given `lm_head` stays unquantized.

### The claim this supports

*A two-round selector, demonstrated on dense Qwen3-8B in vLLM, whose
per-deployment calibration is one singles sweep* — not *these coefficients
describe self-speculative decoding*. The mechanistic findings above are the
portable part, and they are arguably worth more than the coefficients because
they say **why** a re-fit is required rather than merely that it is.

---

## 29. Step 3: the arming case does not reproduce either

Step 3 was planned as "make OFF a first-class candidate", motivated by D3's
R4 regression — 522 against OFF's 684.9, a 24% loss — and by Campaign 1
measuring every armed arm losing to stock at LI and LIO. Reading the phase's
own record before spending GPU time changed the step twice.

**First, the rule already exists.** `w98_failclosed.py` and
`w98_prereg_failclosed.md` define it with no free parameter —
`arm iff margin - 1 > max(EPSILON_ARM, envelope(R))` — with a sealed firing
set `{R4}` and in-sample effect 0.951 -> **0.996** of omniscient. Nothing to
build.

**Second, G98-F already refuted its premise.** On this box R4's armed arm
*wins* at **1.129** against h103's 0.763, with R1 reproducing h103 to 1% as
the control and six independent lines pointing at h103's R4 armed column
having been measured under contention. So the 24% regression this record has
cited repeatedly — including in this document's own plan — **is not
established**, and on this box the rule would *cost* 11% at R4.

That leaves one live question, and it is the one that matters for the grid we
are moving to: **does "speculation loses at long input" reproduce here?**

### The measurement

LI (cap 2K) and LIO (cap 4K), batch 8, quantized family, natural EOS,
registered prompts. `woff/skip0` included deliberately as the most
bandwidth-hungry armed arm — the one G98-F's mechanism would hit hardest.
Two independent boots per cell.

| cell | arm | boot 1 | boot 2 | spread |
| --- | --- | --- | --- | --- |
| **LI** | `woff/skip4` | **1.1552** | 1.1571 | 0.17% |
| | `woff/skip0` | 1.1475 | 1.1489 | 0.12% |
| | `w1024/skip4` | 1.0989 | 1.0939 | 0.45% |
| | `w512/skip4` | 1.0742 | 1.0849 | 0.99% |
| **LIO** | `w1024/skip4` | **1.3626** | 1.3464 | 1.19% |
| | `w512/skip4` | 1.3540 | 1.3382 | 1.17% |
| | `woff/skip4` | 1.2614 | 1.2571 | 0.34% |
| | `woff/skip0` | 1.2510 | 1.2470 | 0.32% |

**Every armed arm beats the parked one at both cells**, and it replicates:
the two OFF boots agree to 0.3% (334.5/333.8 and 374.6/375.8 tok/s), every
armed ratio to within 1.2%. Zero cap hits, token counts within 4% across
arms, so the length confound that dominated LO is small here.

### Against Campaign 1, on the same denominator

Campaign 1 scores against **stock** (plain vLLM), where its `off` arm — our
runtime with speculation parked — is itself 0.933 (LI) and 0.925 (LIO).
Dividing through to armed-over-off on both sides:

| cell | Campaign 1 (h103) | **this box** | discrepancy |
| --- | --- | --- | --- |
| LI b8, `w4a16` | 0.759 | **1.148** | **1.51x** |
| LIO b8, `w1024` | 0.901 | **1.363** | **1.51x** |
| R4 (G98-F) | 0.763 | 1.129 | 1.48x |

**Three independent long-context cells, the same 1.5x factor**, against an
R1 control that reproduces across boxes to 1%. This is the R4 pattern
repeating on the refined grid, and it fits G98-F's stated mechanism: the most
bandwidth-hungry armed steps are the exposed ones, and LI/LIO are exactly
that shape — 9-16K of context, batch 8, an unwindowed draft.

### The verdict, and its limits

**On this box the fail-closed rule has no firing set on the refined grid.**
Every cell measured wants to arm. The rule remains sound and stays in the
code; what is now doubtful is whether it has anything to decline.

What this does **not** establish:

* **Beating OFF is not beating stock.** Campaign 1's `off` sits 6.7-7.5%
  below stock at these cells, which is our runtime's own overhead. Applying
  that factor, our armed arms would score ~1.07 (LI) and ~1.26 (LIO) against
  stock — still wins, but smaller, and taken from another box's overhead
  measurement. **This grid has no stock arm of its own**, and adding one is
  the cheapest thing that would close the gap.
* **A protocol difference exists.** Campaign 1 queues `4 x batch` requests
  behind a concurrency cap; we submit exactly 8 and let them drain (lengths
  331-855 at LI). Real, and not plausibly worth 51%, but not zero.
* **Scope**: one box, one batch, the quantized family, four arms, natural
  EOS. Campaign 1 also measured b16, where its LI/LIO numbers are worse
  (0.615-0.936), and we have not.
* **The reproducibility here is stronger than the phase's standing caveat
  but tests less.** The record's OFF-referenced band is 9-16%; two boots
  minutes apart on a clean box agree to 0.3%. That is evidence the band is
  not active right now, not evidence it does not exist — it was derived over
  a longer horizon and under co-tenancy.

### What it changes

Two of the phase's three "speculation loses" results — R4 and now LI/LIO —
are properties of one machine rather than of the workload. That is a finding
about measurement discipline at least as much as about the selector: **every
one of them came from single boots with no timing gate**, and both times a
two-boot replicate on a clean box reversed the sign.

The consequence for the plan is that step 3 closes without the deliverable it
was scoped for, and the next step is the refined evaluation itself.

---

## 30. The stock arm: section 29's conclusion was drawn against the wrong baseline

Section 29 concluded that "on this box the fail-closed rule has no firing set
on the refined grid", from armed-versus-**OFF** ratios of 1.07-1.16 at LI and
1.25-1.36 at LIO. It also listed, as its first recorded limit, that *beating
OFF is not beating stock* and that the stock-to-off factor had been borrowed
from another box.

That borrowed factor was the thing that mattered. This measures it here.

### The stock arm

Added to `run_w98_refined_lo.py`: plain vLLM, no `speculative_config`, every
`VLLM_SELF_SPEC*` variable stripped from the boot. `w98_artifacts.cell_key`
now names `stock` after itself for the same reason it already named `off` —
same levers, different measurement — so it cannot share a file with the
unlevered armed cell. Two boots per cell.

| | LI | LIO |
| --- | --- | --- |
| stock | 402.8 / 399.0 tok/s | 481.2 / 481.2 tok/s |
| **`off` vs stock** | **0.830 / 0.837** | **0.779 / 0.781** |
| Campaign 1's `off` vs stock (h103) | 0.933 | 0.925 |

**Our runtime, with speculation parked, costs 17% at LI and 22% at LIO on
this box** — against 6.7% and 7.5% on h103. That is a fixed per-step host
tax, which is what the box's clamp is, and our runtime does more host work
per step than stock. It is also 2-3x the tax the borrowed factor assumed.

### Against stock, LI still loses

| cell | arm | boot 1 | boot 2 |
| --- | --- | --- | --- |
| **LI** | `woff/skip4` | **0.959** | **0.968** |
| | `woff/skip0` | 0.953 | 0.961 |
| | `w1024/skip4` | 0.913 | 0.915 |
| | `w512/skip4` | 0.892 | 0.908 |
| **LIO** | `w1024/skip4` | **1.061** | **1.052** |
| | `w512/skip4` | 1.054 | 1.045 |
| | `woff/skip4` | 0.982 | 0.982 |
| | `woff/skip0` | 0.974 | 0.974 |

**Section 29's conclusion is wrong and is corrected here.** Against the
deployment baseline:

* **LI is lost by every arm** (0.892-0.968). Campaign 1's finding that "LI is
  lost by every draft arm" **reproduces in sign**; its magnitude does not
  (0.62-0.78 there against 0.89-0.97 here).
* **LIO is won only by the windowed arms** (1.045-1.061), while the
  unwindowed ones lose (0.974-0.982).

So the fail-closed rule **does** have a firing set on this grid — LI — and
section 29's "every cell wants to arm" was an artifact of measuring against
`off` rather than against stock.

What section 29 got right stands: its armed-versus-`off` ratios reproduce to
1.2%, and the 1.5x discrepancy with h103's *armed* column is real. What it
got wrong is that `off` is not the baseline a deployment compares against,
and on this box the gap between `off` and stock is three times what the
borrowed factor said.

### The selector's job, visible for the first time on this grid

LIO is the cleanest demonstration the phase has produced of why lever
selection matters rather than lever *presence*: at one cell, one batch, the
windowed arms clear stock by 5-6% and the unwindowed arms fall below it. The
same three levers separate winning from losing. That is the selector's
premise measured directly, and it needed the stock arm to be visible at all.

### Two notes on the measurement

**The engine tax is the headline number for honesty.** Every armed result in
this record is a ratio against `off`, and `off` costs 17-22% here. A reader
comparing to a stock deployment must apply that factor, and it is
box-dependent: 6.7-7.5% on bare metal, 17-22% on this clamped guest. Our
arms are therefore *understated* here relative to what h103 would show — but
that is an inference from two h103 numbers, not something this box can claim.

**Stock is not bit-deterministic across boots either.** At LI it emitted
5,653 tokens then 5,458 — 3.5% apart, greedy, same prompts, no speculation at
all. LIO reproduced exactly (9,792 both). Same mechanism as section 22:
batch composition changes reduction order and flips near-tie argmaxes. It is
worth recording that this is not a property of speculative decoding; it is a
property of batched inference.

---

## 31. The refined grid against stock: half of it should not arm

Section 30 added the stock arm at LI and LIO b8. This completes the grid —
b16 for both long-input cells, SS for the first time on this box, and LO's
missing denominator — so every cell is scored against what a deployment
would actually replace.

### The grid

Best arm per cell, against stock:

| cell | batch | best arm | vs stock | `off` vs stock | verdict |
| --- | --- | --- | --- | --- | --- |
| **LI** | 8 | `woff/skip4` | **0.959** | 0.830 | **park** |
| **LI** | 16 | `woff/skip0` | **0.875** | 0.812 | **park** |
| LIO | 8 | `w1024/skip4` | **1.061** | 0.779 | arm |
| LIO | 16 | `w1024/skip4` | **1.224** | 0.892 | arm |
| **SS** | 8 | `woff/skip4` | **0.848** | 0.691 | **park** |
| LO | 8 | `w1024/skip4` | **1.177** | 0.759 | arm |

**Half the grid should not speculate.** LI and SS lose at every arm measured;
LIO and LO win. That is a firing set for the fail-closed rule covering three
of six (cell, batch) points, where section 29 concluded there was none.

### Three findings the grid makes visible

**1. The window lever's sign flips between two long-input cells.** At LI the
windowed arms are the *worst* and get worse with batch — `w1024/skip4` goes
0.913 -> **0.723**, falling below even `off` — while the unwindowed arms lead.
At LIO the windowed arms lead at both batches and their margin *widens*,
1.061 -> **1.224**. Both cells feed the draft 9-16K of context; they differ in
output length (LI 0.3-1K, LIO 1-2K). So "long input" is not the axis that
decides whether a window pays, and a selector keyed on input length alone
would get LI exactly wrong.

**2. Arming with the wrong lever is worse than not arming.** At LI b16 `off`
(0.812) beats `w1024/skip4` (0.723) and `w512/skip4` (0.705); at LIO b16
`off` (0.892) beats `woff/skip4` (0.879). The penalty for a bad lever choice
exceeds the entire engine tax.

**3. The engine tax varies 11-31% by cell**, and it is the dominant term
almost everywhere:

```text
SS b8   0.691      LO b8   0.759      LIO b8  0.779
LIO b16 0.892      LI b16  0.812      LI b8   0.830
```

Every armed number this record has reported against `off` is inflated by that
factor. LO's headline is the clearest case: `w1024/skip4` scores **1.565**
against `off` and **1.177** against stock. Both are correct measurements of
different things, and only the second is what a deployment sees.

### What the selector is worth here

Against the best single static configuration — `w1024/skip4`, which wins
every cell where arming wins at all:

| cell, batch | selector (park or arm) | best static | selector gain |
| --- | --- | --- | --- |
| LI b8 | park, 1.000 | 0.913 | **+9.5%** |
| LI b16 | park, 1.000 | 0.723 | **+38.3%** |
| LIO b8 | arm, 1.061 | 1.061 | 0 |
| LIO b16 | arm, 1.224 | 1.224 | 0 |
| SS b8 | park, 1.000 | 0.824 | **+21.4%** |
| LO b8 | arm, 1.177 | 1.177 | 0 |

**The selector's entire value on this grid is the decision not to arm.**
Where arming wins, one static configuration wins everywhere, so lever
*selection* buys nothing across cells; the arm/park decision buys 9-38% on
three of six points. That is a sharper and less flattering statement than the
R-grid's +1.4% per-regime figure, and it points the design at the gate rather
than at the lattice.

It also vindicates the fail-closed rule's construction. The rule declines
when the predicted margin is smaller than what the cost model can resolve;
here the cells it should decline are the ones where the best arm clears stock
by −4% to −15%, which no envelope in the map could certify as positive.

### Scope, and what is not yet claimed

* **The new points are single boots.** LI b8 and LIO b8 carry replicates
  (agreeing to 1.2%); LI b16, LIO b16, SS b8 and LO stock do not. Given that
  this record has twice been corrected by a replicate, they should get one
  before any of this is used in a claim.
* **LO's arms are natural-EOS** and carry section 25's contamination: at
  equal work `woff/skip8` is last in the lattice, and here it reads second at
  1.169. The LO *ranking* below the top arm should not be read; the top arm
  agrees between protocols.
* **The selector's picks are not modelled here** — this is the measured grid,
  which is the ceiling the selector is scored against. Whether the two-round
  prediction actually declines at LI and SS is the next question, and the
  prediction map for those cells does not exist: their acceptance has never
  been measured, only LO's.
* **One box, and the tax is box-dependent** (6.7-7.5% on h103 against
  11-31% here). Cells whose margin is inside that difference — LI b8 at
  0.959 — could plausibly change sign on bare metal.

---

## 32. The instrument was on our side of every comparison

Section 31 concluded that half the refined grid should not arm, from
best-arm-versus-stock figures of 0.959 (LI b8), 0.848 (SS b8) and 1.061
(LIO b8). Asked to check whether any of the engine tax was removable, the
first place to look was what our boots pay that stock does not.

`run_w98_g98b_round1.boot_environment` sets **`VLLM_SELF_SPEC_PROFILE=1`** and
a **`VLLM_SELF_SPEC_KOFF_TRACE`** path on every boot it builds — `off` and
every armed arm alike. The stock arm added in section 30 has neither, because
it was constructed by stripping all `VLLM_SELF_SPEC*` variables. Amendment 2
measured the profiler's inner syncs at **+2.4 ms/step**, and the trace writes
a JSON line per step on top of that. Phase 100's own campaign runner turns
both off and says why: *"Profiler and koff trace OFF -- wall clock is the
score and the instrument costs +2.4 ms/step."*

So every armed number in sections 29-31 was measured against a baseline that
did not pay the instrument.

### Re-measured with the instrument off

Same cells, same arms, same protocol, `VLLM_SELF_SPEC_PROFILE=0` and no trace:

| cell | arm | instrument ON | **OFF** | gain |
| --- | --- | --- | --- | --- |
| **LI** | `woff/skip4` | 0.9592 | **1.0632** | +10.8% |
| | `w1024/skip4` | 0.9125 | **1.0574** | +15.9% |
| | `woff/skip0` | 0.9529 | **1.0034** | +5.3% |
| | `off` | 0.8304 | **0.9154** | +10.2% |
| **LIO** | `w1024/skip4` | 1.0609 | **1.2323** | +16.2% |
| | `woff/skip0` | 0.9741 | **1.0381** | +6.6% |
| | `off` | 0.7786 | **0.9301** | +19.5% |
| **SS** | `woff/skip4` | 0.8482 | **1.0545** | +24.3% |
| | `w1024/skip4` | 0.8237 | **1.0471** | +27.1% |
| | `woff/skip0` | 0.7915 | **1.0112** | +27.8% |
| | `off` | 0.6913 | **0.7513** | +8.7% |

**Section 31's park verdicts are withdrawn.** With the instrument removed,
**every cell wins**: LI 1.063, LIO 1.232, SS 1.055. There is no firing set on
this grid after all — which is where section 29 started, though for a reason
neither section had.

### The confirmation that the diagnosis is right

The engine tax, `off` against stock:

| cell | instrument ON | **OFF** | Campaign 1 (h103) |
| --- | --- | --- | --- |
| LI | 0.8304 | **0.9154** | 0.933 |
| LIO | 0.7786 | **0.9301** | 0.925 |

**Instrument-free, our tax matches h103's to within 2%** — 0.915 against
0.933, 0.930 against 0.925 — where instrumented it was off by 10-15 points.
Two campaigns on different boxes now agree on the parked overhead, which is
what a correct diagnosis of an instrument artifact should produce, and it was
not a fit: nothing here was tuned to make those numbers meet.

SS is the exception and keeps a **24.9%** tax. Its sequences are short, so a
fixed per-step overhead is amortised over the fewest tokens of any cell. That
is the constant term behaving exactly as a constant term should.

### What survives, and what does not

* **The armed discrepancy with h103 shrinks but does not close.** Their `w4a16`
  at LI b8 scores 0.708 against our comparable `woff/skip0` at 1.0034 — a
  factor of **1.42**, down from section 29's 1.51 but still large. The
  instrument explains the *tax* term completely and the *armed* gap only
  partly.
* **Section 31's structural findings stand**: the window lever's sign still
  flips between LI and LIO, arming with the wrong lever is still worse than
  parking, and the engine tax is still cell-dependent. Only the park/arm
  verdicts move.
* **Section 30 stands entirely.** Measuring against stock rather than `off`
  was necessary and remains so; it was the *instrumented* stock comparison
  that was wrong, not the choice of baseline.

### The pattern, which is the durable part

This is the **third** time in this record a comparison was contaminated by
something one side paid and the other did not:

| section | what one side paid | size |
| --- | --- | --- |
| 21, 25 | different generation lengths under natural EOS | up to 22 points |
| 30 | the runtime's own overhead, via an `off` denominator | 17-22 points |
| **32** | **the measurement instrument** | **5-28 points** |

Each was found by the same question — *what does this baseline pay that the
other does not?* — and each reversed a published conclusion. The question is
cheap and should be asked before any cross-arm number is recorded, not after
it is contradicted.

### Scope

Single boots per arm at three cells, batch 8, quantized family. LI b16, LIO
b16 and LO have not been re-measured instrument-free, and their section-31
numbers should be read as instrumented until they are.

---

## 33. Instrument-free: the window's "sign flip" was the instrument too

Section 32 re-took LI, LIO and SS at batch 8 without the profiler or the koff
trace and withdrew section 31's park verdicts. It left LI b16, LIO b16 and LO
instrumented, so their published numbers were still scored against a baseline
that did not pay the instrument. This completes them.

### LI b16, where the effect is largest in the record

| arm | instrumented | **instrument-free** | gain |
| --- | --- | --- | --- |
| `w1024/skip4` | 0.7234 (worst, below `off`) | **1.1281 (best)** | **+56.0%** |
| `w512/skip4` | 0.7046 | **1.1122** | **+57.8%** |
| `woff/skip4` | 0.8653 | 0.9873 | +14.1% |
| `woff/skip0` | 0.8754 | 0.9719 | +11.0% |
| `off` | 0.8117 | 0.9194 | +13.3% |

**The instrument did not shift the arms uniformly — it reordered them.** The
windowed arms gain 56-58% where everything else gains 11-14%, which takes
them from the bottom of the cell (below the parked arm) to the top (above
stock).

**Section 31's first finding is withdrawn.** "The window lever's sign flips
between two long-input cells" was an artifact: instrument-free, `w1024/skip4`
is the *best* arm at LI b16, as it already was at LIO and LO. There is no
flip.

The mechanism is one this record already measured. Section 26 established
that the window's value is **(attention time removed) − (host time exposed)**,
and the window path is the one that does extra host work per step —
`_apply_draft_kv_window` rebuilds the block table and sequence lengths on
every draft step. The profiler's inner syncs inflate exactly that term, so
the arms carrying the most host work are penalised most. An instrument that
costs host time reorders levers that differ in host work, and this is that
happening.

### LIO b16 and LO

| cell | arm | instrumented | **instrument-free** |
| --- | --- | --- | --- |
| LIO b16 | `w1024/skip4` | 1.2243 | **1.3314** |
| | `off` | 0.8920 | 0.9441 |
| | `woff/skip4` | 0.8786 | 0.9312 |
| LO b8 | `w1024/skip4` | 1.1774 | **1.3583** |
| | `woff/skip8` | 1.1687 | 1.2947 |
| | `off` | 0.7588 | 0.8560 |

**Section 31's second finding survives, narrowly.** At LIO b16 `off` (0.9441)
still beats `woff/skip4` (0.9312) and `woff/skip0` (0.9339), so arming with
the wrong lever remains worse than not arming — but by 1%, at one cell,
rather than the 8-12% across two that the instrumented numbers showed.

### The corrected grid

Best arm against stock, instrument-free, natural EOS, quantized family:

| cell | batch | best arm | vs stock | `off` vs stock |
| --- | --- | --- | --- | --- |
| LI | 8 | `woff/skip4` | **1.063** | 0.915 |
| LI | 16 | `w1024/skip4` | **1.128** | 0.919 |
| LIO | 8 | `w1024/skip4` | **1.232** | 0.930 |
| LIO | 16 | `w1024/skip4` | **1.331** | 0.944 |
| SS | 8 | `woff/skip4` | **1.055** | 0.751 |
| LO | 8 | `w1024/skip4` | **1.358** | 0.856 |

**Every cell wins, and `w1024/skip4` wins four of six** — the two exceptions
being the cells with the shortest generations, where `woff/skip4` leads.

The engine tax, instrument-free, runs **5.6% to 24.9%** and orders itself the
way a fixed per-step cost should: worst at SS (0.751), whose sequences are
shortest; next at LO (0.856), whose batch drains from 8 active to 1 so late
steps amortise over fewer tokens; and mildest at LI/LIO b16 (0.919-0.944),
which hold the most tokens per step. That is the constant term the search
strategy should carry, and it is a per-step cost divided by tokens per step,
not a flat percentage.

### What still does not close

The armed discrepancy with h103 is unmoved by any of this: their `w4a16` at
LI b8 reads 0.708 against our comparable `woff/skip0` at 1.0034, a factor of
**1.42**. The instrument explained the *parked* tax completely — two boxes now
agree to 2% — and explains none of the armed gap.

### Scope

Single boots per arm. LO carries only four arms (parked, unlevered, deep
skip, top pick) because its generations are ~100x longer than LI's, and its
arms remain natural-EOS with section 25's contamination below the top arm.

---

## 34. The bf16 family, and a contamination the replicates caught

Two jobs section 33 left open, both instrument-free: the bf16 family at the
three cells where every armed arm so far has been quantized, and replicates
of the instrument-free grid.

### The denominators check out first

| cell | stock, bf16 run | stock, quantized run | apart |
| --- | --- | --- | --- |
| LI | 398.9 | 399.3 | **0.09%** |
| LIO | 481.1 | 481.4 | 0.08% |
| SS | 710.9 | 711.4 | 0.06% |

And the parked arm agrees between weight versions to **0.25-0.63%** (LI
0.9211 vs 0.9154, LIO 0.9263 vs 0.9301, SS 0.7532 vs 0.7513), reproducing at
three new cells what section 15 found at LO: the parked path is genuinely
quant-independent, because the draft is resident but never runs. The two
families sit in one table by right.

### Quantization is load-bearing, not a default we inherited

Against stock, instrument-free, batch 8:

| cell | arm | bf16 | w4a16 | quant buys |
| --- | --- | --- | --- | --- |
| **LI** | `woff/skip4` | **0.9252** | **1.0632** | +14.9% |
| | `w1024/skip4` | 0.8778 | 1.0574 | +20.5% |
| | `woff/skip0` | 0.7960 | 1.0034 | +26.1% |
| | `w512/skip4` | 0.8461 | — | — |
| **LIO** | `w1024/skip4` | 1.0060 | **1.2323** | +22.5% |
| | `w512/skip4` | 1.0030 | — | — |
| | `woff/skip0` | 0.9750 | 1.0381 | +6.5% |
| | `woff/skip4` | 0.9106 | 0.9903 | +8.8% |
| **SS** | `woff/skip4` | **0.7869** | **1.0545** | +34.0% |
| | `w1024/skip4` | 0.7764 | 1.0471 | +34.9% |
| | `woff/skip0` | 0.7717 | 1.0112 | +31.0% |

**Quantization wins every arm at every cell, by 6.5% to 34.9% — and at LI and
SS it is the difference between losing and winning.** With the bf16 draft
every arm at LI (0.796-0.925) and SS (0.772-0.787) falls below stock; with
the quantized draft every one clears it.

So "quantize the draft" is not a convenience generalized from LO. On two of
the three cells measured here it decides whether speculation is worth doing
at all, and a selector restricted to the bf16 family would correctly park at
LI and SS — the firing set sections 29-33 have been arguing about exists, but
only for the *other* weight version.

The size also tracks traffic share the way section 27's account predicts, with
SS highest: SS removes ~72% of draft traffic (negligible KV, weights dominate)
against LO's 54.5%, LIO's 39.2% and LI's 35.4%.

### The replicates caught a contamination, with a timestamp

Twenty-five arms compared across six (cell, batch) points. **Twenty-three
reproduce to within 1.5%.** Two did not:

| arm | boot 1 | boot 2 | spread |
| --- | --- | --- | --- |
| `lio_b8 / woff/skip0` | 1.0381 | **0.5442** | **47.6%** |
| `li_b16 / w1024/skip4` | 1.1281 | 1.0459 | 7.3% |

Both are explained by the same event, and the evidence is direct rather than
inferred. A co-tenant process appeared mid-run and **OOM'd one of our boots at
11:35:12** (`Process 2900885 has 4.11 GiB memory in use`). The 47.6% outlier
was written at **11:36:55** — immediately after, with the co-tenant still
resident. The 7.3% outlier landed at 11:33:38, ninety seconds before. Both
read slow, and no arm outside that window moved more than 1.5%.

**This is the h103 mechanism, reproduced on our own box with a clock.** G98-F
attributed h103's R4 anomaly — and sections 29 and 31 the LI/LIO discrepancy —
to contention on unreplicated single boots, on six lines of circumstantial
evidence. Here the same failure produced a 47.6% error, and the only reason
it did not become a finding is that the boot was replicated. It is also a
direct answer to the question of whether that mechanism is real on this
hardware: it is, it is large, and it is invisible in a single boot.

The two contaminated records are kept rather than overwritten, and a third
replicate was measured on a verified-quiet box (memory below 2 GB before
launch). It closes the diagnosis:

| arm | boot 1 | boot 2 (contended) | **boot 3 (quiet)** |
| --- | --- | --- | --- |
| `lio_b8 / woff/skip0` | 1.0381 | 0.5442 | **1.0536** |
| `li_b16 / w1024/skip4` | 1.1281 | 1.0459 | **1.1295** |

Both clean boots agree with boot 1 -- to 1.5% and 0.12% -- so boot 2 is
isolated as the outlier rather than the pair being averaged. The first
attempt at boot 3 itself failed with a CUDA OOM during graph capture while
another tenant held memory, which is its own measurement of how often this
machine is contended.

### What this does to the grid

Nothing, for the quantized family: `lio_b8 / woff/skip0` and
`li_b16 / w1024/skip4` are neither cell's best arm, so section 33's headline
figures — LI b8 1.063, LI b16 1.128, LIO b8 1.232, LIO b16 1.331, SS 1.055,
LO 1.358 — all replicate to within 0.5%, and the best arm at every cell
reproduces to **0.15% or better**.

### Scope

The bf16 family is single boots at three cells; only the quantized family is
replicated. `w512/skip4` has no quantized twin at these cells, so its rows
are bf16-only. LO's bf16 family is not re-measured here — section 25 covers
it at equal work.

---

## 35. Acceptance on LI, LIO and SS: the window's real price, and an inflation confirmed

Twenty-one boots, the quantized family's seven arms across the three cells
whose acceptance had never been measured. Generation budgets matched to each
cell's natural output (LI 1024, LIO 2048, SS 512), registered u-edges
unchanged, `ignore_eos` per the calibration protocol. All 21 succeeded.

`tau_k4`, re-derived at the deployed depth from `pos_accepted`:

| arm | LI b0 | LI b1 | LIO b0 | LIO b1 | LIO b2 | SS b0 | SS b1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `woff/skip0` | **4.448** | **4.542** | **4.305** | **4.345** | 4.808 | 4.806 | 5.000 |
| `woff/skip4` | 4.254 | 4.334 | 4.000 | 3.931 | 4.724 | 4.779 | 5.000 |
| `woff/skip8` | 3.453 | 3.568 | 3.075 | 3.014 | 4.105 | 4.293 | 4.969 |
| `w1024/skip4` | 2.920 | 3.308 | 3.252 | 3.294 | 4.425 | 4.779 | 5.000 |
| `w512/skip4` | 2.892 | 3.043 | 3.242 | 3.105 | 4.300 | 4.759 | 5.000 |
| `w256/skip4` | 2.855 | 2.818 | 3.171 | 2.962 | 4.105 | 4.797 | 4.973 |
| `w128/skip4` | 2.772 | 2.570 | 3.053 | 2.807 | 3.597 | 4.587 | 4.288 |

### The window's price is enormous at long input, and zero at short

At LI the window costs **31%** of acceptance — `woff/skip4` 4.254 against
`w1024/skip4` 2.920 — and at LIO **19%**. That is the opposite end of the
scale from LO, where the same window barely moved acceptance, and the reason
is geometric: LI feeds the draft 12,109-19,147 tokens of context, so a
1024-token window shows it **6%** of the document it is summarizing.

At SS the window costs **nothing**: `w1024/skip4` and `woff/skip4` both read
**4.779**, identical to three decimals. SS prompts are 41-145 tokens, so a
1024 window cannot bind and the two arms are the same configuration. That
identity is a control the campaign passed without being asked to.

**This is what sections 31-34 could not see.** They measured `w1024/skip4` as
the best arm at LI b16 (1.128) and LIO (1.232, 1.331) — and acceptance now
says the window arms have the *worst* acceptance at exactly those cells. So
the window wins on throughput **despite** losing a third of its acceptance,
because at 12-17K of context the KV traffic it removes outweighs the tokens
it gives up. That is a genuine cost-acceptance trade, and it is the first one
on this grid the two-round design has both halves of.

### The `ignore_eos` inflation is real, and now measured

Earlier in this arc the researcher predicted it: with `ignore_eos`, once a
request passes its natural stopping point the continuation is trivial filler
that the draft predicts perfectly, inflating acceptance. This measures it.

SS's natural output is **median 153, p95 350** tokens. Bucket 1 covers
u >= 256, so it is almost entirely **past** natural EOS — and there four of
seven arms read **`tau_k4` = 5.000**, which at K=4 is *every drafted token
accepted, every step*. Perfect acceptance is not a property of the workload;
it is the signature of predicting filler.

The same effect is visible wherever a bucket runs past a cell's natural
length:

| cell | natural median / p95 | bucket past it | what it reads |
| --- | --- | --- | --- |
| SS | 153 / 350 | b1 (256+) | **5.000** for four arms |
| LIO | 1,257 / 1,459 | b2 (1024-3072) | 4.1-4.8, above b0/b1 for **every** arm |
| LI | 771 / 999 | b1 partly (256-1024) | mildly above b0 |

**Registered consequence: an acceptance curve must be truncated at the
cell's natural output distribution before it is fed to a prediction.** SS's
bucket 1 and LIO's bucket 2 are calibration artifacts, not workload
acceptance, and using them would over-predict every arm — most at the cells
where generations are shortest. The researcher's split stands and is now
quantified: `ignore_eos` is right for calibration and must not be read past
the length the scored protocol would have stopped at.

### What this unblocks

The prediction map for LI, LIO and SS is now computable: cost coefficients
per family (section 24), acceptance per cell (here), and the length
distributions the parked arms measured (section 21's registered input). The
selector can finally be **scored** against section 33's ceiling rather than
handed it.

### Scope

Quantized family only — section 34 established it as the deployed default
and showed the bf16 family losing to stock at LI and SS, so predicting the
bf16 arms is needed only to verify the selector rejects them. Single boots,
but acceptance is near-deterministic at fixed realization: G98-F measured
bit-identical accept patterns across boxes, which is also why these two cells
were safe to run concurrently on separate GPUs.

---

## 36. The prediction map for the new cells: SS works, LI and LIO fail for a located reason

With section 35's curves the prediction map is computable for all four cells.
Scored against the instrument-free measurements, quantized family, batch 8,
curves truncated at each cell's natural p95 length per section 35:

| cell | mean abs error | predicted best | measured best |
| --- | --- | --- | --- |
| **SS** | **0.0164** | `w1024/skip4` | `woff/skip4` |
| LO (section 25) | 0.0366 | `w1024/skip4` | `w1024/skip4` |
| **LIO** | **0.2254** | `woff/skip0` | `w1024/skip4` |
| **LI** | **0.3223** | `woff/skip4` | `woff/skip4` |

**SS predicts to 1.6%**, and its rank swap is unresolvable — 1.421 against
1.368 predicted, 1.403 against 1.394 measured. **LI and LIO over-predict by
23-32%.**

### It is not the KV superlinearity

The obvious suspect was extrapolating a linear KV term 3.5x past its
calibration, so section 13's `gamma` was swept:

| gamma | LI | LIO | SS |
| --- | --- | --- | --- |
| 1.00 | 0.3223 | 0.2254 | 0.0164 |
| 1.15 | 0.3222 | 0.2298 | 0.0169 |
| 1.25 | 0.3219 | 0.2323 | 0.0170 |

Nothing moves. **Refuted**, and the reason is that the KV term is not merely
wrongly scaled — it has the wrong sign.

### It is section 24's unidentified window term, extrapolated

The quantized family's per-family fit returned `kappa_kv = -2.96e-12` and
`f_win = -4.56 ms`, both unphysical. Section 24 diagnosed why: at LO's
context the window-size axis carried no resolved signal (`g256 - g1024` =
0.315 +/- 0.165 ms, 1.9 sigma), so the bytes form had nothing to fit but
noise. Section 24 also recorded the consequence as bounded — "the honest form
predicts only at windows that were measured".

It is not bounded when the *context* moves. Evaluating that fit:

| | `woff`, keep 0.889 | `w1024` | `w256` |
| --- | --- | --- | --- |
| at LO calibration (156+8192) | 24.57 ms | 21.78 | 22.06 |
| **at LI (14611+999)** | **20.35 ms** | 21.76 | 22.06 |

**The model says reading 15,000 KV positions is 4.2 ms cheaper than reading
4,252.** A negative per-byte coefficient is harmless where KV is small and
absurd where it dominates, and the error's shape is exactly that signature:
the two **unwindowed** arms over-predict by **+40.8%** and **+42.0%**, the
windowed one by **+13.9%**, because only the unwindowed arms collect the
spurious discount.

The three cells then order themselves by how much KV the draft reads:

| cell | mean KV positions | prediction error |
| --- | --- | --- |
| SS | ~200 | **1.6%** |
| LO | 4,252 | 3.7% |
| LIO | ~12,900 | 22.5% |
| LI | ~15,100 | 32.2% |

Prediction quality is a function of distance from the calibration context,
and the failure is not in the acceptance measured in section 35 — SS and LO
use the same pipeline and land inside 4%.

### What this settles about the search strategy

Round 1's premise is that composed cells can be predicted from single-lever
profiles. Sections 25 and here confirm it **within the context the profiles
were fitted at**, and this is the first clean demonstration of where it stops:
a coefficient that is unidentified at calibration context is not merely
imprecise elsewhere, it is **unbounded**, because nothing constrains its sign.

The registered fix is a cost calibration at long context: the equal-work
sweep uses 156-token prompts, so `kappa_kv` and `f_win` have never been
fitted where they dominate. An LI-shaped equal-work sweep -- three windows at
three keeps on 12-16K prompts -- identifies them where the refined grid
actually operates, and is the same nine boots per family that section 24's
sweep was.

### Scope

Three arms per cell, quantized family, batch 8. LI and LIO carry `w512` and
`w128` in the acceptance campaign but not in the throughput grid, so only the
three arms measured both ways are scored.

---

## 37. The long-context cost calibration: signs fixed, ranks fixed, one bug named

Section 36 registered the fix — fit `kappa_kv` and `f_win` where KV dominates
— and this runs it. Three windows at three keeps on LI's 12,160-token
prompts, 1,024 generated, quantized family complete at nine arms; the bf16
path trims to four on non-LO cells by design, which tests transfer rather
than fitting a second lattice.

**The sweep waited for the box.** A first attempt was launched while another
tenant held 77 GB on GPU 0 at 100% util; five boots OOM'd and **no record was
written**, which is the benign failure — a boot that dies is visible, one
that runs 40% slow is not. It was re-queued behind a gate requiring the whole
box under 2 GB across two samples 60 s apart, and fired at 15:49 on 0 MiB
with no other tenants.

### What the LO-fitted model was doing at LI

| arm, keep 1.0 | LO fit predicts | **measured** | error |
| --- | --- | --- | --- |
| `woff` | 22.99 ms | **44.942 ms** | **−48.9%** |
| `w1024` | 23.50 ms | 23.760 ms | −1.1% |
| `w256` | 23.84 ms | 23.581 ms | +1.1% |

The windowed arms predict to **1.1% at a context 3x beyond calibration**,
because a window saturates and bounds its own KV. The unwindowed arm is
under-charged by half. That is the whole of section 36's 32% error, and it
is one arm's cost term.

The window's worth, by context:

```text
at LO (ctx 4252):   off 26.659 -> w1024 23.513 =  3.15 ms
at LI (ctx 12672):  off 44.942 -> w1024 23.760 = 21.18 ms
```

**A 6.7x swing that the LO sweep could not see**, because at LO the window
removes a quarter of the KV and at LI it removes 92% of it.

### Refitting across both contexts

Eighteen arms spanning 156 to 12,160 prompt tokens, residual 0.0265:

| coefficient | LO-only (section 24) | **two-context** |
| --- | --- | --- |
| `kappa_kv` | **−2.96e−12** | **+1.395e−11** |
| `f_win` | **−4.56 ms** | **+4.12 ms** |
| `c_layer` | — | 12.24 ms |
| `F` | — | 6.03 ms |

**Every coefficient is physical for the first time in the quantized family.**
The fitted `kappa_kv` also lands within 35% of the bf16 family's independent
LO value (1.035e−11), where before the two had opposite signs.

Re-scoring the prediction:

| cell | LO-only fit | **two-context fit** | ranking |
| --- | --- | --- | --- |
| **LI** | 0.3223 | **0.1675** | **exactly correct** |
| **LIO** | 0.2254 | **0.0942** | **exactly correct** |
| SS | 0.0164 | 0.1186 | degraded |

LI and LIO now rank all three arms in the measured order, which neither did
before. The residual error is one-sided — every arm over-predicted by
8-18% — which is a level effect, not a ranking one, and the selector consumes
the ranking.

### The SS regression names a specification bug

At SS the model computes **250 KV positions for `woff` and 250 for
`w1024`**: with 75-token prompts and 350 generated, a 1024 window has nothing
to truncate. The KV term is therefore identical for the two arms — and the
model still charges the windowed one `f_win`, now +4.12 ms, roughly 18% of an
SS draft chain.

Measurement says the window is free there. Section 35 already showed it:
`w1024/skip4` and `woff/skip4` have **identical acceptance to three
decimals** (4.779 both), because they are the same configuration. Throughput
agrees — 1.394 against 1.403, **0.6% apart**. The model puts them 9% apart
and ranks the window last.

**`f_win` is charged as `[w > 0]` where the data supports `[w > 0 and the
window actually binds]`.** It was invisible until now because every cell
previously fitted or scored had a context far exceeding its windows. The
LO-only fit hid it a second way: with `f_win` negative, the windowed arm got
a spurious *bonus* that happened to cancel the missing gate, which is why SS
scored 0.0164 on a model that was wrong twice.

### Where this leaves the search strategy

Round 1's premise — composed cells predicted from single-lever profiles —
now holds across a 60x context range on one coefficient set, with correct
ranking at the two cells that previously failed. What it took was calibrating
where the dominant term dominates, and the general statement is section 36's:
a coefficient unidentified at calibration context is unbounded elsewhere,
because nothing constrains its sign.

The remaining defects are both specification rather than measurement: the
unbinding window gate above, and the one-sided 8-18% level offset at LI/LIO.

---

## 38. Calibrate at the long end: it transfers down, not up

Two changes tested against section 37's open items — the `f_win` binding gate
it named, and the choice of calibration context it raised.

### The binding gate: right, and not the fix

`w98_cost_u.window_binds` now charges `f_win` only where
`window + sinks < context`, at all three call sites. It is correctly scoped:
**LI and LIO are unchanged to four decimals**, because their windows bind. At
SS it does what it was designed to do and removes the phantom charge — and
the prediction gets *worse*, 0.1186 to 0.1755.

So the gate is right on the physics (section 35 measured those two SS arms as
literally the same configuration, identical acceptance to three decimals) and
was never the cause of SS's error. Kept, because a model that charges for a
lever that does nothing is wrong whether or not that error happens to
dominate.

### The cause was the calibration context

Three cost fits, each scored against all three cells:

| fit | LI | LIO | SS |
| --- | --- | --- | --- |
| LO-only (prompt 156) | 0.3224 | 0.2216 | **0.0164** |
| **LI-only (prompt 12,160)** | **0.1645** | **0.0848** | **0.0241** |
| both, pooled | 0.1675 | 0.0942 | 0.1755 |

**Calibration at the long end transfers down; calibration at the short end
does not transfer up.** The LI-fitted coefficients hold at a cell with 60x
less context (SS, 2.4%) while the LO-fitted ones fail by 32% in the other
direction.

The reason is identifiability, and it is section 36's principle stated
positively. At long context the KV term *varies* across the window axis —
`woff` reads 12,672 positions and `w1024` reads 1,040 — so `kappa_kv` is
pinned by the design. At short context every arm reads ~250 positions
whatever its window, so `kappa_kv` is fitted on nothing, and a coefficient
fitted on nothing is not merely imprecise away from home: it is unbounded,
because nothing constrains its sign. Extrapolating a *pinned* coefficient
into a regime where its term is small is safe; extrapolating an unpinned one
into a regime where its term is large is not.

**The pooled fit is worse than the long-only fit at every cell**, including
the two it was meant to serve. Pooling two contexts with different residual
structure degrades both rather than averaging them, so "fit everywhere at
once" is not the resolution either.

### Where the selector now stands

Long-context fit, quantized family, curves truncated at natural length:

| cell | mean abs error | predicted rank | measured rank |
| --- | --- | --- | --- |
| LI | 0.1645 | `woff/skip4 > w1024/skip4 > woff/skip0` | **identical** |
| LIO | 0.0848 | `w1024/skip4 > woff/skip0 > woff/skip4` | **identical** |
| SS | **0.0241** | `w1024 > woff/skip4 > woff/skip0` | top two swapped |
| LO | 0.0366 | `w512 > w1024 > w256 > w128 > ...` | top two swapped |

**Two cells rank exactly; the other two swap a pair that measurement does not
separate** — SS's top two differ by **0.6%** measured (1.403 against 1.394)
and LO's by 1.2%, both inside this box's armed-arm stability. On the ranking
the selector actually consumes, one coefficient set now covers the whole
refined grid.

The residual is a one-sided level offset: every LI and LIO arm is
over-predicted by 8-17%, in the same direction, which moves no ranks. It is
the last unexplained term and it is not what the selector reads.

### Registered guidance

**Calibrate the cost model at the longest context the deployment will see.**
The equal-work sweep's 156-token prompts were chosen when LO was the only
cell; on a grid spanning 75 to 19,147 prompt tokens that choice put every
coefficient's identifiability at the wrong end. Nine boots at the long end
buy the whole range.

---

## 39. The selector scored end to end: the prediction works, the selection is worth 0.15%

Every input now exists, so this runs the selector's own decision rule rather
than describing it. Picks are the argmax of **predicted** value from the
long-context fit (section 38); the measured grid supplies only the realized
rate, so the selector is free to be wrong. Rates are against each cell's own
**stock** boot, instrument-free. Four cells, four arms common to all of them,
equal token mix, time-weighted.

### Per cell

| cell | selector picks | rate | omniscient | rate | share |
| --- | --- | --- | --- | --- | --- |
| LI | `woff/skip4` | 1.0632 | `woff/skip4` | 1.0632 | **100%** |
| LIO | `w1024/skip4` | 1.2323 | `w1024/skip4` | 1.2323 | **100%** |
| LO | `w1024/skip4` | 1.3583 | `w1024/skip4` | 1.3583 | **100%** |
| SS | `w1024/skip4` | 1.0471 | `woff/skip4` | 1.0545 | 99.29% |

**Three of four picks are the omniscient pick.** The one miss is SS, where
the two arms differ by 0.7% measured — the near-tie section 38 already
identified as unresolvable, and section 35 explained: at 75-token prompts a
1024 window cannot bind, so those two arms are close to the same
configuration.

### The mix

```text
selector      1.1617 of stock
omniscient    1.1639
              99.80% of omniscient
```

| static (one arm everywhere) | rate |
| --- | --- |
| **`w1024/skip4`** | **1.1599** |
| `woff/skip4` | 1.0889 |
| `woff/skip0` | 1.0642 |
| `woff/skip8` | 1.0441 |
| `off` (parked) | 0.8571 |

**The selector reaches 99.80% of omniscient and beats the best single static
by 0.15%.**

### Both halves of that sentence matter, and they are different claims

**The prediction is validated.** One coefficient set, calibrated at one
context, ranks four cells spanning 75 to 19,147 prompt tokens well enough to
pick the omniscient arm three times out of four and lose 0.7% on the fourth.
That is the two-round design's central premise — cost predictable offline,
acceptance measured where it must be — and on this grid it holds.

**The selection is worth nothing.** `w1024/skip4` held constant scores 1.1599
against the selector's 1.1617. There is no cell on this grid where the right
lever is meaningfully different, so per-cell selection has nothing to buy.

This is the Phase-82 dwell-time law and D3's frontier finding arriving a third
time, now on the refined grid against a stock baseline: *switching pays only
where the mix makes cells differ enough, and on this workload family they do
not.* D3 measured the selector beating the best static by 1.4% on the
R-grid and failing its +2% clause; here it is 0.15%.

### What the value actually is

Not selection between levers, but the two decisions either side of it:

* **arming at all** is worth **+35.5%** over the parked runtime (1.1617
  against 0.8571) and **+16.2%** over stock;
* **not picking a bad lever** is worth **+11.3%** (best static 1.1599
  against worst 1.0441).

A deployment that picks `w1024/skip4` once and never thinks again gets
essentially everything. The selector's machinery earns its place by finding
that arm and by being right about *why* — which is what makes it transfer to
a grid where the answer is not constant — not by switching on this one.

### Scope

Equal token mix over four cells at batch 8, quantized family, single boots for
the four arms added to complete the common set. Section 33 measured LIO's
margin widening with batch (1.232 -> 1.331 from b8 to b16) while LI's best arm
changes identity, so a batch-swept mix is where a selection case would have to
come from; it is not measured here.

---

## 40. The batch-swept grid: the switching case exists, and the selector captures 12% of it

Section 39 found selection worth **+0.15%** over the best static, because one
arm sat at or near the top of all four cells. Section 33 named batch as the
axis on which the optimum actually moves. This sweeps it: eight (cell, batch)
points, four arms common to every one, instrument-free, against each point's
own stock boot.

### The optimum does move

| point | `w1024/skip4` | `woff/skip0` | `woff/skip4` | `woff/skip8` | best |
| --- | --- | --- | --- | --- | --- |
| LI b8 | 1.0574 | 1.0034 | **1.0632** | 0.9654 | `woff/skip4` |
| LI b16 | **1.1281** | 0.9719 | 0.9873 | 0.9232 | `w1024/skip4` |
| LIO b8 | **1.2323** | 1.0381 | 0.9903 | 0.9381 | `w1024/skip4` |
| LIO b16 | **1.3314** | 0.9339 | 0.9312 | 0.8216 | `w1024/skip4` |
| SS b8 | 1.0471 | 1.0112 | **1.0545** | 1.0451 | `woff/skip4` |
| **SS b32** | 1.1310 | 1.1064 | **1.3440** | 1.0471 | **`woff/skip4`** |
| LO b8 | **1.3583** | 1.2348 | 1.2907 | 1.2947 | `w1024/skip4` |
| LO b16 | **1.6637** | 1.1834 | 1.1742 | 1.1665 | `w1024/skip4` |

**SS b32 is the separation the grid was missing**: the unwindowed arm beats
the window by **19%** where at SS b8 they were 0.7% apart. And LI's best arm
changes identity between batches, as section 33 said it would.

Neither static is good everywhere. `w1024/skip4` gives up 16% at SS b32;
`woff/skip4` gives up 30% at LIO b16 and 29% at LO b16.

### The selector, scored on the swept mix

| point | selector | rate | omniscient | rate | share |
| --- | --- | --- | --- | --- | --- |
| LI b8 | `woff/skip4` | 1.0632 | `woff/skip4` | 1.0632 | 100% |
| **LI b16** | `woff/skip4` | **0.9873** | `w1024/skip4` | **1.1281** | **87.52%** |
| LIO b8 | `w1024/skip4` | 1.2323 | same | 1.2323 | 100% |
| LIO b16 | `w1024/skip4` | 1.3314 | same | 1.3314 | 100% |
| **SS b32** | **`woff/skip4`** | **1.3440** | same | 1.3440 | **100%** |
| SS b8 | `w1024/skip4` | 1.0471 | `woff/skip4` | 1.0545 | 99.29% |
| LO b8 | `w1024/skip4` | 1.3583 | same | 1.3583 | 100% |
| LO b16 | `w1024/skip4` | 1.6637 | same | 1.6637 | 100% |

```text
selector     1.2209 of stock
omniscient   1.2462          -> 97.97%
best static  1.2174 (w1024/skip4 everywhere)
             selector beats it by +0.29%
```

**The model caught the hard one.** SS b32 is the largest lever reversal on
the grid — 19% — and the prediction picks it correctly, having been
calibrated at a completely different context. That is the two-round design
doing exactly what it exists for.

**And it lost the value again on LI b16**, where it predicts `woff/skip4`
(1.743) over `w1024/skip4` (1.654) while measurement puts them at 0.987
against 1.128. That single miss costs 12.5% of one point, and time-weighted
aggregation punishes low rates hardest, so it very nearly cancels the SS b32
win.

### The number that matters

```text
omniscient over best static   +2.37%     <- the switching case that EXISTS
selector   over best static   +0.29%     <- the part the selector CAPTURES
                                            = 12% of what is available
```

**Sweeping batch produced a real switching case where the equal-batch grid had
none** — the ceiling rises from ~0.3% to **+2.37%** — and the selector
realizes an eighth of it. So the honest reading of sections 39 and 40
together is not "switching is worthless" but **"switching is worth ~2% here
and our selector is not yet accurate enough to collect it."** Those are very
different conclusions and only the second is actionable.

### The suspect for LI b16, untested

The acceptance curves are measured at **batch 8** and applied at 16 and 32.
Section 22 measured acceptance moving with concurrent batch by *different
amounts per lever* — 5.9% for deep skip against 2.8% for the window arm — and
sized `tau(u, B)` as "real but modest, likely small". LI b16 is the first
place in this record where that gap could cost something, and the miss has
the right shape: the model over-rates the unwindowed arm at the batch where
the window's advantage is widening.

That is a hypothesis. It was not tested here, and the alternative -- a cost
term that scales wrongly with batch at long context -- is equally live, since
the cost fit is also batch-8.

### Scope

Single boots at the four new points, equal token mix over eight points,
quantized family. Verify cost at b16 and b32 is extrapolated from section
20's affine fit rather than measured at those batches.

---

## 41. LI b16: the acceptance hypothesis is refuted, and the miss is a cost effect

Section 40 named `tau(u, B)` as the suspect for the selector's one damaging
miss — LI b16, where the model predicts `woff/skip4` over `w1024/skip4` while
measurement puts them at 0.987 against 1.128 — and marked it untested. This
tests it: the same two arms, same LI content, same 1024-token budget, at
**batch 16** instead of 8, so the only variable that moves is B.

### Acceptance barely moves

| arm | b8 bucket 0 | b16 bucket 0 | b8 bucket 1 | b16 bucket 1 |
| --- | --- | --- | --- | --- |
| `woff/skip4` | 4.254 | 4.188 | 4.334 | 4.302 |
| `w1024/skip4` | 2.920 | 2.969 | 3.308 | 3.172 |

The quantity the ranking depends on is the ratio between them:

```text
bucket 0:  b8 1.457 -> b16 1.411   (-3.2%)
bucket 1:  b8 1.310 -> b16 1.356   (+3.5%)
```

**Under 4% in both buckets, and in opposite directions.** That is section 22's
non-monotone batch effect at its measured size, and it is nowhere near enough
to reverse a 31% acceptance gap. Feeding the b16 curves into the prediction
makes it *worse* — mean error 0.4610 against 0.4022 with the b8 curves, rank
still wrong.

**`tau(u, B)` is refuted as the cause.** Section 22's own sizing — "real but
modest, likely small" — holds, and section 40's suspicion of it was wrong.

### The cost side is where the arms separate

Measured per-token cost, the two arms against each other:

| | `woff/skip4` | `w1024/skip4` | ratio |
| --- | --- | --- | --- |
| b8 | 2.356 ms | 2.369 ms | **0.995** |
| b16 | 2.055 ms | 1.799 ms | **1.143** |

**At batch 8 the two arms cost the same to within 0.5%; at batch 16 the
unwindowed arm costs 14.3% more.** Acceptance held fixed, so the entire
reversal is cost.

The mechanism is the one the model should already have: the draft chain reads
KV once per request per forward, so the window's saving scales with batch.

```text
B=8   woff reads 71.3 GB/step, w1024 4.91 GB  -> difference 66.4 GB = 19.8 ms at peak
B=16  woff reads 142.6 GB/step, w1024 9.81 GB -> difference 132.8 GB = 39.6 ms at peak
```

The model does carry `keep * KV * B`, so it doubles that difference correctly.
What it does not carry is why doubling it changes the *ranking*: at b8 the
window's KV saving is already large enough that both arms are limited by
something else, and at b16 it is not. Section 26 measured that "something
else" directly — the window's value is (attention time removed) minus (host
time exposed), and only the first term scales with batch.

**So the defect is section 26's non-separability, evaluated at a second
batch.** The window's overhead `f_win` is fitted once, at batch 8, and
charged flat; its benefit scales with B. A term that is constant in B
competing with a term linear in B cannot rank correctly at both ends unless
the constant is right, and section 37 fitted it at one end only.

### What it costs, and what it does not

This is the whole of the selector's loss on the swept grid: correcting LI b16
alone moves the selector from 1.2209 to 1.2462 — **the omniscient value** —
turning +0.29% over the best static into the full **+2.37%**. One arm at one
batch is the entire gap between capturing an eighth of the switching case and
capturing all of it.

It is also a narrow defect rather than a broad one. Seven of eight points are
already exact, the cost model ranks correctly at three other cells and at
batch 32, and the fix is a fitted quantity rather than a missing mechanism:
`f_win` measured at two batches instead of one.

### Scope

Two arms, one cell, one batch pair. The b16 acceptance is a fresh measurement;
the cost comparison uses the section-40 throughput records. Whether `f_win`
refitted across batches actually corrects the rank is untested — this locates
the term, it does not repair it.

---

## 42. `f_win` refitted across batch: the rank is corrected, and it was not `f_win`

Section 41 located the selector's one damaging miss in a cost term fitted at
a single batch and left the repair untested. This runs it: the same nine-arm
equal-work design on LI prompts at **batch 16**, giving 18 arms across two
batches.

### The measurement the batch-8 fit could not see

| arm, keep 1.0 | b8 | **b16** | growth |
| --- | --- | --- | --- |
| `woff` | 44.942 ms | **74.526 ms** | **+65.8%** |
| `w1024` | 23.760 ms | 25.186 ms | +6.0% |
| `w256` | 23.581 ms | 23.495 ms | −0.4% |

Doubling batch costs the unwindowed draft 30 ms and the windowed one 1.4 ms.
That is the whole of section 41's reversal, measured directly at the draft
chain rather than inferred from throughput.

### `f_win` is batch-shared, and my reading of the code was wrong

Three forms fitted on all 18 arms, `nnls`-constrained:

| form | `f_win` | mean residual |
| --- | --- | --- |
| A: flat `keep*[w>0]` | 5.215 ms shared | **2.471%** |
| B: per-request `keep*[w>0]*B` | 0.086 ms/request | 2.962% |
| C: both, jointly | **5.215 ms shared, 0 per-request** | 2.471% |

Form C drives the per-request column to **exactly zero**. I expected the
opposite: `_apply_draft_kv_window` opens with `bs = cad.num_reqs` and compacts
every request's page list, so the compaction plainly does work proportional
to batch. **The data says that work is not on the critical path** — it is
metadata manipulation over a few hundred bytes per request, hidden behind
kernels that read gigabytes.

So section 41's diagnosis was right about *where* (a cost term fitted at one
batch) and wrong about *which*. `f_win` never needed a batch axis.

### What did need it: `kappa_kv` as a per-request coefficient

The correction is the `--batched-fit` column — KV bytes charged per step as
`keep * kv * B` rather than per sequence — which makes `kappa_kv` a true
per-request coefficient rather than one that silently absorbs the fitting
batch. That distinction is invisible in a single-batch design, and every cost
arm in this phase before now shared one batch.

With 18 arms across two batches under that form:

| point | before | **after** |
| --- | --- | --- |
| **LI b16** | `woff/skip4` (**wrong**) | **`w1024/skip4`** (correct) |
| LI b8 | `woff/skip4` | `woff/skip4` (unchanged, all four arms ranked exactly) |

### The selector, re-scored on the swept grid

| point | selector | rate | omniscient | share |
| --- | --- | --- | --- | --- |
| LI b8 | `woff/skip4` | 1.0632 | same | 100% |
| **LI b16** | **`w1024/skip4`** | **1.1281** | same | **100%** |
| LIO b8 | `w1024/skip4` | 1.2323 | same | 100% |
| LIO b16 | `w512/skip4` | 1.2995 | `w1024/skip4` | 97.61% |
| SS b8 | `w1024/skip4` | 1.0471 | `woff/skip4` | 99.29% |
| SS b32 | `woff/skip4` | 1.3440 | same | 100% |
| LO b8 | `w1024/skip4` | 1.3583 | same | 100% |
| LO b16 | `w1024/skip4` | 1.6637 | same | 100% |

```text
                          section 40    section 42
selector of stock            1.2209        1.2414
share of omniscient          97.97%        99.61%
over best static             +0.29%        +1.97%
switching case available     +2.37%        +2.37%
captured                        12%           83%
```

**The selector now captures 83% of the switching case, against 12% before.**

### The new miss is a different failure, and a milder one

LIO b16 picks `w512/skip4` (1.2995) over `w1024/skip4` (1.3314) — 2.4% apart,
a within-window-family choice rather than a lever reversal. Both windowed
arms beat every unwindowed arm at that point by 39%, so the selector gets the
*lever* right and the *size* wrong. Section 24 predicted exactly this: the
window-size axis is the one it could not resolve (`g256 - g1024` at 1.9
sigma), and it remains the weakest direction in the design.

### Honest accounting of this sequence

Sections 40-42 were three hypotheses, and the record should show the score:

| section | hypothesis | verdict |
| --- | --- | --- |
| 40 | `tau(u, B)` — acceptance moves with batch | **refuted** (§41: under 4%, both directions) |
| 41 | `f_win` fitted at one batch | **located the area, named the wrong term** |
| 42 | `kappa_kv` absorbing the fitting batch | **confirmed** — rank corrected, b8 unchanged |

The productive pattern was not the hypotheses but the discriminator: each was
tested by a measurement that could return either answer, and two returned the
answer that killed them.

### Scope

Nine boots at b16, quantized family, LI content. The two-batch fit is
validated by the rank it corrects and by leaving b8 unchanged; it is not
validated at batch 32, where SS's prediction still uses it and lands correctly
but is a single point.

---

## 43. The window-size axis, measured: resolved in cost, and the miss survives

Section 42 left one miss — LIO b16 picking `w512/skip4` over `w1024/skip4` —
and section 24 had already named window size as the design's weakest
direction (`g256 - g1024` at 1.9 sigma). `w512` was in neither long-context
cost sweep, so its cost was interpolated across exactly that unresolved gap.
Six boots close it: `w512` at three keeps, at b8 and b16, on LI prompts.

### The axis resolves at b16 and still does not at b8

Draft chain, keep 1.0:

| window | b8 | b16 |
| --- | --- | --- |
| `off` | 44.942 | 74.526 |
| `w256` | 23.581 | **23.495** |
| `w512` | **23.033** | **23.926** |
| `w1024` | 23.760 | **25.186** |

**At b16 the window-size axis is finally ordered by bytes** — 23.495 <
23.926 < 25.186, monotone, spanning 7.2%. At b8 it is not: `w512` measures
*cheaper* than both its neighbours, a non-monotone pattern inside the ~1%
noise, which is section 24's 1.9 sigma reproducing at a longer context.

So the axis needed **both** a long context and a large batch to separate. That
is a third instance of this record's recurring lesson, and the sharpest: the
term was unidentified not because the design lacked levels, but because the
*operating point* made its differences small.

### The miss survives anyway

Refitting on 24 arms with `w512` measured — residual improves 2.471% to
2.218% — and re-scoring:

| point | selector | omniscient | share |
| --- | --- | --- | --- |
| LIO b16 | `w512/skip4` 1.2995 | `w1024/skip4` 1.3314 | **97.61%** |

Unchanged. The grid score is unchanged too: **99.61% of omniscient, +1.97%
over the best static.** Measuring the interpolated arm did not move the
decision, so **the miss is not a cost-interpolation artifact**, which is what
this sweep was run to find out.

### What the miss actually is

Both inputs are now measured and both are nearly tied:

```text
cost      w512 21.913 ms against w1024 23.158     -> w512 cheaper by 5.4%
acceptance  bucket 0  1.003    bucket 1  1.061    -> w1024 higher by 0-6%
measured    w512 1.2995 against w1024 1.3314      -> w1024 wins by 2.4%
```

The model has `w512` ahead by 1.0%; measurement has `w1024` ahead by 2.4%. It
is wrong by **3.4 points on a near-tie between two arms of the same lever
family**, where the replicate spread on this grid is 1.2% and the model's own
level error is 8-25%. Both windowed arms beat every unwindowed arm at that
point by 39%, so the selector picks the lever correctly and the size
marginally wrong.

**This is where the design's resolution ends rather than a defect with a
next fix.** Section 24 said the window-size axis was the weakest direction;
sections 37, 42 and 43 have now each improved a coefficient without moving
it, and the remaining error is smaller than the gap between the two arms
being ranked.

### Where the arc stands

| | value |
| --- | --- |
| selector, swept grid | **1.2414 of stock** |
| share of omniscient | **99.61%** |
| over the best single static | **+1.97%** |
| switching case available | +2.37% |
| **captured** | **83%** |
| points picked exactly | **6 of 8** |

The two misses are a 2.4% near-tie (LIO b16) and a 0.7% near-tie (SS b8),
both between arms whose separation is at or below this box's replicate
spread.

### Scope

Six boots, single, quantized family. The b8 non-monotonicity is asserted as
noise on the basis of its size relative to replicate spread, not on repeated
measurement of those three arms.

---

## 44. Replicating the swept grid: 36 of 38 arms hold, and the two that move are both baselines

Every point in sections 40-43 was a single boot, and this record has been
corrected by a replicate three times. This replicates the four swept points
that had none, each with its own fresh stock boot.

**Thirty-eight arms compared across eight (cell, batch) points.** Excluding
the two co-tenant contaminations section 34 already diagnosed and confirmed
with a third boot, **every armed arm reproduces to 1.31% or better** — and at
the six long-generation points, to 1.19%.

### The exception: SS b32's denominator, not its arms

Every arm at SS b32 moved by ~20%, uniformly:

| arm | boot 1 | boot 2 | spread |
| --- | --- | --- | --- |
| `woff/skip4` | 1.3440 | 1.0606 | 21.1% |
| `w1024/skip4` | 1.1310 | 0.9016 | 20.3% |
| `woff/skip0` | 1.1064 | 0.8746 | 21.0% |
| `off` | 0.8120 | 0.6489 | 20.1% |

A uniform shift in every ratio is a denominator effect, and it is:

| boot | tokens | wall | **longest request** |
| --- | --- | --- | --- |
| stock | 5,476 | 3.123 s | **454** |
| stock r2 | 5,388 | 2.446 s | **350** |
| `off` | 5,476 | 3.846 s | 454 |
| `off` r2 | 5,476 | 3.831 s | 454 |

**In absolute tok/s the armed arms reproduce to 0.87%** (`woff/skip4` 2357.0
against 2336.4; `w1024/skip4` 1982.8 against 1986.0; `off` 1423.7 against
1429.4). Only stock moved.

The mechanism is section 30's finding meeting section 21's. Stock is not
bit-deterministic across boots — greedy fixes the rule for choosing a token,
not the logits — and at SS b32 that non-determinism moved the **longest**
request from 454 tokens to 350. The run is 2.4-3.1 seconds with a batch that
drains to one, so the tail request sets the wall clock: a 23% shorter tail is
a 26% faster run.

**This is a short-cell measurement defect, not a lever effect.** It is worst
exactly where sections 21 and 35 said length effects bite hardest — the cell
with the shortest generations — and the registered remedy is Campaign 1's:
`n = max(16, 2b)` requests rather than `n = batch`, so no single tail
dominates. We submit `batch` requests, which at b32 is 32 against Campaign
1's 64.

### What it does to the claims, and what it does not

Re-scoring the whole grid under each denominator:

| | boot 1 stock | boot 2 stock at SS b32 |
| --- | --- | --- |
| selector, of stock | 1.2414 | **1.2042** |
| share of omniscient | **99.61%** | **99.62%** |
| over the best single static | **+1.97%** | **+2.30%** |

**The two claims the phase makes are insensitive to it, and the third is
not.** Share-of-omniscient and over-best-static both compare arms measured
against the *same* denominator, so it cancels; the absolute "x stock" figure
does not, and carries roughly 3% uncertainty in the mix from this one cell.

Reported accordingly: **99.61-99.62% of omniscient and +1.97% to +2.30% over
the best static**, with the absolute multiple over stock stated as ~1.20-1.24
rather than to four figures. The ranking at SS b32 is identical in both boots
(`woff/skip4 > w1024/skip4 > woff/skip0 > woff/skip8 > off`), so the
selector's pick there — the largest lever reversal on the grid — is not in
question.

### The two known contaminations, closed

| arm | boot 1 | boot 2 | boot 3 (quiet box) |
| --- | --- | --- | --- |
| `lio_b8 / woff/skip0` | 1.0381 | 0.5442 | **1.0536** |
| `li_b16 / w1024/skip4` | 1.1281 | 1.0459 | **1.1295** |

Both third boots agree with boot 1, isolating the contended measurements
rather than averaging them. Section 34's diagnosis stands: a co-tenant OOM'd
one of our boots at 11:35:12 and the outliers were written at 11:36:55 and
11:33:38.

### Scope

Two boots at each of eight points, one third boot at each of the two
contaminated arms. SS b32's instability is diagnosed from two boots and its
mechanism is inferred from the request-length records, not from a repeated
measurement of the tail.

---

## 45. The full lattice: no static policy is best, and the earlier grid could not have shown it

Sections 39-44 scored the selector against a "best static" chosen from four
arms common to every point — of which exactly **one** carried a window. Five
of the twenty window x skip combinations per weight version were ever
measured. So `w1024/skip4` won that comparison by being the only windowed arm
at cells where windows dominate, and the claim that no static policy is best
was not tested.

This tests it. **588 boots**: 40 lattice configurations (both weight versions
x 5 windows x 4 skips) plus `off` and `stock`, at all 14 registered
`(cell, batch)` points, instrument-free, natural EOS, ~22 GPU-hours.

### The grid

Best arm at each point, against that point's own stock boot:

| point | best arm | vs stock | `off` vs stock |
| --- | --- | --- | --- |
| LI b1 | `woff/skip4` | 1.1616 | 0.851 |
| LI b8 | **`w512/skip4`** | 1.0921 | 0.922 |
| LI b16 | **`w512/skip4`** | 1.1552 | 0.916 |
| LIO b1 | `woff/skip0` | 1.1472 | 0.848 |
| LIO b8 | **`w512/skip4`** | 1.2515 | 0.925 |
| LIO b16 | `w1024/skip4` | 1.3302 | 0.944 |
| LO b1 | `woff/skip4` | 1.3556 | 0.850 |
| LO b8 | **`w1024/skip8`** | **1.6595** | 0.854 |
| LO b16 | **`w256/skip0`** | **1.7096** | 0.908 |
| LO b32 | **`w512/skip4`** | **1.7471** | 0.927 |
| SS b1 | `woff/skip8` | **0.9703** | 0.849 |
| SS b8 | `woff/skip4` | 1.0677 | 0.753 |
| SS b32 | `woff/skip4` | 1.0679 | 0.630 |
| SS b64 | `woff/skip4` | 1.0320 | 0.797 |

**Seven distinct arms win across fourteen points.** `woff/skip4` (5),
`w512/skip4` (4), and one each for `woff/skip0`, `w1024/skip4`,
`w1024/skip8`, `w256/skip0`, `woff/skip8`.

### No static policy is best — measured

| candidate static | mix | worst point |
| --- | --- | --- |
| **`w512/skip4`** | **1.1412** | **−24.8%** |
| `w1024/skip4` | 1.1307 | −24.6% |
| `w1024/skip0` | 1.0940 | −26.2% |
| `woff/skip4` | 1.0782 | −39.0% |

* **The best static is not the one the earlier grid found.** `w512/skip4`
  displaces `w1024/skip4` — an arm the four-arm set carried at only two of
  fourteen points.
* **Every candidate gives up 25-39% at its worst point.** With the four-arm
  set the figure was −15.8%.
* **The omniscient mix is 1.2243 against the best static's 1.1412, so a
  single fixed configuration gives up 6.79%.** Section 40 measured that gap
  at 2.37% on the impoverished set. **Filling the lattice nearly tripled the
  switching case**, because it added arms that are much better *somewhere*
  without being better everywhere.

### The composed arms are the top of the lattice, as D2(a) predicted

At LO b8 the top three are `w1024/skip8` (1.6595), `w512/skip8` (1.6266) and
`w256/skip8` (1.6161) — **windowed deep skip, all three**. The earlier grid's
best static ranks **7th of 40** there, at 1.3614, beaten by **21.9%**.

That is D2(a)'s constructive composition measured end to end: a window that
has already discarded the context removes the very information layer skipping
would have degraded, so the two levers' losses overlap instead of compounding.
`woff/skip8` alone is mediocre; *windowed* deep skip is the best arm on the
highest-value cell.

Two more lattice facts fall out:

* **`skip16` never places in a top-5** at any of the fourteen points, and
  occupies the bottom of LO b8 (0.596-0.642). That is D2(b)'s additivity
  boundary — the knapsack's ranking inverts at k=16 — showing up end to end.
* **bf16 appears in 2 of 70 top-5 slots and wins nothing.** Every one of the
  fourteen winners is `w4a16-quantized`, confirming section 34's default on
  the full lattice rather than on a sample.

### A fail-closed firing set exists, and it is one point

**SS b1: the best arm measures 0.9703 — no configuration beats stock.** Every
other point has a winning arm. So the rule has exactly one place to fire on
this grid, at the cell with the shortest sequences and the smallest batch,
where a fixed per-step cost is amortised over the fewest tokens.

### What this does to the earlier claims

Sections 39-44's *measurements* stand — they are a subset of this grid and
reproduce within it. Their *interpretation* does not:

| claim | on four arms | **on the full lattice** |
| --- | --- | --- |
| distinct winning arms | 3 of 8 points | **7 of 14 points** |
| best static's worst point | −15.8% | **−24.8%** |
| switching value available | +2.37% | **+6.79%** |
| best static's identity | `w1024/skip4` | **`w512/skip4`** |

The selector's own score is not restated here: its prediction map covers the
five arms the earlier grid measured, not forty, so scoring it on this lattice
requires acceptance for the thirty-five arms that have none. That is the next
measurement, and it is now the only thing between this grid and an end-to-end
number.

### Scope

Single boots per arm; the earlier grid's replication (section 44) found armed
arms reproducing to 1.31% and one baseline unstable at SS b32. Fourteen points
at one box, quantized and bf16 families, natural EOS, decode currency.

---

## 46. The engine tax, located: one synchronization per step

Sections 30-33 measured our runtime parked (`off`) against plain vLLM
(`stock`) and found a cell-dependent tax of 5.6-37%, worst where steps are
cheapest. The researcher's position was that this is implementation overhead
rather than an intrinsic cost of having speculation available, and that if it
were removed the two baselines would converge. This investigates it.

Two candidates were eliminated before profiling, from data the phase already
owned:

* **HBM residency.** `off` measures within **0.20-0.54%** of itself whether
  the draft is a separate 6.1 GB resident (`w4a16-quantized`) or shares the
  target's own tensors (`target-matching`, `SHARE_WEIGHTS=1`). A cost that
  does not move when 6.1 GB appears is not a memory cost.
* **The instrument.** Section 32 removed the profiler and koff trace and the
  tax fell from 17-22% to 5.6-9.4% at LI/LIO, with the residual reproducing
  on h103 (0.933/0.925 against our 0.915/0.930). What remains is our
  implementation, not this box.

### The device does identical work

`off` pins K to 0, so it should run exactly the target's forward passes. It
does, to 0.07%:

| | stock | `off` |
| --- | --- | --- |
| SS b32 device | 7.290 ms/step | **7.285** |
| LO b8 device | 6.981 ms/step | **6.966** |
| SS b32 wall | 11.910 | **14.946** |
| LO b8 wall | 11.869 | **14.575** |
| SS b32 **idle** | 4.619 | **7.661** |
| LO b8 **idle** | 4.906 | **7.627** |

**The entire tax is idle time.** The GPU does the same work and waits longer
for the host.

### It is one call

Ranked by per-step host delta, the top entry is not merely largest — it is
almost the whole thing:

| | stock | `off` | share of tax |
| --- | --- | --- | --- |
| SS b32 `cudaEventSynchronize` | 3.368 ms, 1.0 calls | **6.046 ms, 2.0 calls** | **88%** |
| LO b8 `cudaEventSynchronize` | 3.637 ms, 1.0 calls | **5.801 ms, 2.0 calls** | **80%** |

Every other host event moves by under 70 us/step, and several move the other
way. **`off` performs one additional blocking synchronization per step, and
it costs 2.2-2.7 ms.**

### Where it comes from, and why it is structural

`gpu_model_runner.py:942-945` creates `num_accepted_tokens_event` whenever
`self.num_spec_tokens` is set, and `:2107` synchronizes on it every step:

```python
if self.num_spec_tokens:
    self.draft_token_ids_event = torch.Event()
    self.num_accepted_tokens_event = torch.Event()
...
if self.num_accepted_tokens_event is not None:
    self.num_accepted_tokens_event.synchronize()
```

The condition is `num_spec_tokens`, which is the **configured maximum** (K=4
in our boots), not the per-step schedule. `off` sets
`num_speculative_tokens_per_batch_size: [[1, 32, 0]]` — zero tokens drafted —
but `num_speculative_tokens` remains 4, so the event is created and
synchronized on every step even though no acceptance count is ever produced.

**This is upstream vLLM's speculative-decoding path, not phase-98 code.** Any
deployment that configures speculation and then disables it per-batch pays
it, which is exactly what a K/OFF ladder does whenever it parks.

### What this settles

* **The tax is not intrinsic to having speculation resident.** It is a
  host-side round-trip whose only purpose is to read back a count that a
  parked step does not compute.
* **It is a fixed per-step cost**, which is why the *relative* tax tracks
  step cheapness: 25.5% of a stock step at SS b32 and 22.8% at LO b8 in
  absolute terms, but 37% and 14.6% of throughput because SS b32's steps are
  cheap and LO b8's are not.
* **The researcher's premise is supported.** A gate on the per-step schedule
  rather than the configured maximum would remove 80-88% of the residual tax
  and bring `off` close to `stock` — the condition under which scoring against
  `off` and against `stock` converge.

### What it does NOT settle, and the honest caveat

**A fix is not demonstrated here.** The measurement locates the call and
attributes the cost; it does not show that gating it is correct. The event is
also recorded unconditionally at `:1605` and `:1611`, and whether the
synchronize can be skipped when zero tokens were drafted requires reading the
async-scheduling path that consumes `num_accepted_tokens`, not just the call
site. That is a change to upstream runtime code and belongs behind a test,
not behind this measurement.

**And it does not transfer to the armed arms unchanged.** An armed step
genuinely produces acceptance counts, so it needs the readback; the question
there is whether it can be overlapped rather than removed. Section 26 already
measured that the armed path's host time is what suppresses lever value — the
window's benefit is (attention removed) minus (host exposed) — so removing
host stalls would raise armed throughput *and* reorder levers, as section 33
demonstrated when the instrument came out. **The tax is not a uniform factor
that cancels in a ratio.**

### Scope

Two operating points, single boots, `torch.profiler` with CPU and CUDA
activities, 40 profiled steps after 30 warmup. The profiler inflates absolute
host time, so the wall figures here are larger than the scored runs' and only
the stock-versus-off comparison within each profile is used.
