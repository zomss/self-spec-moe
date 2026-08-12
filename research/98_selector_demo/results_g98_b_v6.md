# G98-B v6 — Round 1 re-measured on piecewise + Marlin, amendment 1 in force

Date: 2026-08-13

**Coverage 24/42 overall; 16/28 over resolvable regimes.** R5 and R5cot are
reported NOT RESOLVABLE under amendment 1 section 5. Round 1 (v5, Machete,
fit-residual-only envelope) was 21/42.

Non-scored for D1: `d1_exercised` remains false, because W14/D still has no
scored surface. This reports coverage and grants no Round-2 authority.

## The envelope, and which term binds

| regime | fit_term | sigma_repro | 2*sigma | envelope | resolvable |
| --- | --- | --- | --- | --- | --- |
| R1 | 0.02541 | 0.00690 | 0.01381 | 0.05785 | yes |
| R4 | 0.00930 | 0.00299 | 0.00598 | 0.02210 | yes |
| **R5** | 0.01052 | 0.00686 | 0.01372 | 0.03458 | **NO** |
| **R5cot** | 0.01161 | 0.00608 | 0.01215 | 0.03362 | **NO** |
| R6 | 0.01942 | 0.00159 | 0.00319 | 0.03937 | yes |
| R8 | 0.02360 | 0.00307 | 0.00614 | 0.04878 | yes |

## The bracketing protocol changed the answer, and inverted the prediction

X15's feasibility estimate (2 anchors, 3 interleaved repeats over ~30 minutes)
predicted R1 would be the marginal regime. The operative `sigma_repro`, measured
from anchors bracketing the real ~2-hour campaign, says otherwise:

| regime | X15 (30 min) | v6 (bracketed, 2 h) | change |
| --- | --- | --- | --- |
| R1 | 0.01410 | 0.00690 | 2.0x better |
| R5 | 0.00178 | 0.00686 | **3.9x worse** |
| R5cot | 0.00261 | 0.00608 | **2.3x worse** |

The long-context regimes are 2-4x noisier over a full campaign than over a short
interleave. This is exactly the drift amendment 1 section 3 requires bracketing
to capture: had X15's numbers been used as operative, R5 and R5cot would have
been scored as covered when the instrument cannot resolve them.

## Where the remaining misses are

All 12 misses on resolvable regimes are **one-directional** — measured above the
band, none below — so the model remains conservative in the direction that
matters for the elimination rule.

| cell family | covered (resolvable regimes) |
| --- | --- |
| target-matching (bf16) | **11/12** |
| w4a16 quantized | **5/16** |
| — of which `woff` | 4/4 |
| — of which windowed | **1/12** |

**The factored model composes correctly for bf16 levers and under-predicts
quantized composed configurations.** The single bf16 miss is 1.012x. The
quantized windowed cells miss by 1.006-1.194x.

## The anomaly shrank but did not vanish

| | quant x skip8 miss ratios |
| --- | --- |
| v5 (Machete, piecewise) | 1.187 – 1.524 |
| **v6 (Marlin, piecewise)** | **1.045 – 1.194** |

Roughly two thirds of the original quant×skip8 penalty was the host-bound
runtime, and the kernel change removed it. The remaining 5-19% is a genuine
composition effect that survives on a runtime where the host is no longer the
bottleneck, and it is specific to **quant × window** (the `woff` quantized cell
covers 4/4 while the windowed quantized cells cover 1/12).

That is a narrower and better-posed question than the one this phase started
with, and it is the one Round 2 should be built around.

## What this does not establish

* D1 is not exercised. Coverage is reported; the false-elimination count under
  `(K+1)/q_lo < 1 + epsilon_arm` is not computed, so it is not claimed.
* The scored runtime is not the fastest known. Piecewise+Marlin trails
  whole-chain by 1.21x at batch 1 and ~1% at batch >= 32 (X14). Option A was
  chosen because the frozen lattice cannot support a whole-chain fit, not
  because piecewise is better.
* Two of six regimes are unresolvable at this precision. Reducing
  `sigma_repro` at long context — or accepting a larger lattice to raise the fit
  term — is a prerequisite for scoring them.

## Artifacts

* `data/w98_g98b_authorization_v6.json`
* `data/g98_b_v6/{round1_result,d1_fits,d1_predictions}.json`
* `data/g98_b_v6/{anchors_pre,singles,heldout,anchors_post}/` — 21 boots
