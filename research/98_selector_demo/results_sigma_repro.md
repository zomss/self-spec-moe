# X15 — Option A is resolvable: sigma_repro on piecewise + Marlin

Date: 2026-08-13
Status: complete. **All six regimes pass amendment 1 section 5.** Option A
(re-measure the lattice on piecewise) is viable, with R1 marginal.

## Result

Two `woff` fit-set anchors, three interleaved repeats each, `draft_chain`
measured through the Round-1 harness unmodified. Boot-to-boot CV:

| regime | baseline anchor | quant anchor |
| --- | --- | --- |
| R1 | 0.027% | **1.413%** |
| R4 | 0.180% | 0.224% |
| R5 | 0.087% | 0.178% |
| R5cot | 0.261% | 0.090% |
| R6 | 0.080% | 0.185% |
| R8 | 0.128% | 0.502% |

Section-5 test, `sigma_repro` = max across anchors, Z = 2:

| regime | sigma_repro | 2*sigma | Round-1 fit_term | resolvable |
| --- | --- | --- | --- | --- |
| R1 | 0.01410 | 0.02820 | 0.0300 | yes (6% margin) |
| R4 | 0.00224 | 0.00447 | 0.0068 | yes |
| R5 | 0.00178 | 0.00355 | 0.0095 | yes |
| R5cot | 0.00261 | 0.00523 | 0.0074 | yes |
| R6 | 0.00185 | 0.00371 | 0.0208 | yes |
| R8 | 0.00502 | 0.01004 | 0.0262 | yes |

## Marlin is what changed this

Round 1's piecewise+Machete reproducibility was CV 1.70% (`sigma` 0.0169,
`2*sigma` 0.0337), which exceeded **every** fit term — that campaign was not
resolvable. With Marlin, `sigma` falls 3-10x and lands below all of them.

This is the same mechanism X14 identified: Marlin issues fewer host launches, so
it helps most exactly where the runtime is host-bound. It cut the batch-1 bubble
from 9.51 ms to 3.12 ms, and the boot-to-boot spread fell with it. **The kernel
choice did not merely make the runtime faster; it made the experiment
measurable.**

## Two caveats that bound this result

**R1 is marginal** at 0.0282 against 0.0300 — a 6% margin — and the two anchors
disagree by 50x there (0.027% vs 1.413%). At batch 1 with short context the
draft chain is smallest and the quantized path is most host-bound, so R1 is the
noisiest cell in the lattice and the one most likely to flip.

**These are Round 1's fit terms, measured with Machete.** The re-measurement
produces new ones and section 5 must be applied against those. This is a
feasibility check that Option A is worth running, not a verdict on D1.

**And this is not the operative `sigma_repro`.** Amendment 1 section 3 requires
repeats bracketing the campaign; there is no campaign yet. The operative value
gets measured around the real run and may differ.

## Decision

Proceed with **Option A**: re-measure the 15 lattice configurations on
piecewise + Marlin under the frozen preregistration, with anchors bracketing the
campaign, then fit and score D1 with the amended envelope.

The lattice blocker does not apply — piecewise carries no `window > 0`
constraint, so the skip and quant axes are sampled on the same runtime as
everything else and the fit is identifiable.

Cost of choosing A over B: the scored runtime trails whole-chain by 1.21x at
batch 1 and ~1% at batch >= 32 (X14). That is the honest price of keeping the
frozen lattice and avoiding a new preregistration.
