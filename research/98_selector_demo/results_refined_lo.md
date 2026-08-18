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
