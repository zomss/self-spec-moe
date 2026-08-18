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
