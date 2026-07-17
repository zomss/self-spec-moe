# Phase 86 results — the KnapSpec head-to-head at Qwen3-8B

## Pre-registered predictions: P1-P3 all CONFIRMED

- **P1 (QK-norm rule, predictive)**: kvq_fp8 = 0.9852 on QK-normed
  Qwen3-8B vs Qwen2.5-7B's broken 0.60-0.84. F6 called a new dense model
  in advance.
- **P2 (skip family-dependence)**: Qwen3-8B far friendlier than Qwen2.5
  (skip125 .754 vs .448; leave-one-out .90-.95; greedy frontier
  .951/.923/.885/.849/.813/.773/.694 at budgets 1-7) -- yet skip still
  does not enter the 8B map: skip-alone prices 0.90x, skip x W4 1.16x,
  both below quant-led.
- **P3 (match at 8B) -- EXCEEDED**: see below.

## The head-to-head table (Qwen3-8B, ours measured vs their published)

| setting | nospec | ours (config) | ratio | KnapSpec |
|---|---|---|---|---|
| b1/16k greedy | 135.0 | **192.0 (W4+win512, fixed chain)** | **1.42x** | 1.28x |
| b1/16k T=0.7 | 134.6 | 177.8 (same) | **1.32x** | (their sampled setting) |
| b1/2k greedy | 148.9 | 183.8 (W4 alone; window moot at 2k) | **1.23x** | ~their short-ctx regime |
| b8/16k | 620.6 | 1124.7 (W4+win, K4) | **1.81x** | no counterpart |
| b16/16k | 838.3 | 1503.7 (W4+win, K6) | **1.79x** | no counterpart |
| b32/16k | 854.4* | 2001.7 | 2.34x* | *nospec capacity-capped (Qwen3-8B KV 147KB/tok; b16 nospec ~= b32 nospec) -- flagged, not headline |

Accept lengths 4.31-5.78 across arms; beta transferred from the offline
column at every cell (e.g. W4 4.311 vs tau(0.952,4)=4.55 minus chain
effects).

## Reading

1. **At their scale and setting we BEAT, not match** (1.42x vs 1.28x
   greedy; 1.32x vs 1.28x sampled) -- and the margin IS the thesis: W4
   alone delivers 1.08x at b1/16k (floor-taxed piecewise chain), their
   lever compositions price <=1.16x (our measured frontier), but
   quant x window x floor-free-chain -- three components of this work
   composed -- clears both. No single lever gets there.
2. **Short context is the honest boundary**: at b1/2k the window is
   moot and W4-alone (1.23x) is the map's answer -- comparable to their
   number, not above it. Composition pays where regimes give it room
   (ctx, batch); exactly the regime-dependence claim.
3. **The batch rows (1.8x) have no counterpart in their evaluation** --
   the map covers their regime AND ours.
4. Cross-system caveat (stated in the paper): both sides report
   speedup-vs-own-AR-baseline; absolute systems differ.
5. b1/2k composed scratchpad arm: engine error (window stack at
   ctx<=2k) -- irrelevant to the map (window is not selected at 2k);
   backlog note.

Next: E3 = the same column at Qwen3-32B (W4-GPTQ ckpt to build), where
P4 predicts composed ~1.5-1.6x vs their 1.43x.

# E3/E4 — the Qwen3-32B column: P4 CLOSED AS A BEAT (task-matched)

Beta column (16k, 1152 pos): win512 .9714, kvq .9878 (QK-norm again),
q_fp8 .9922, skip125 .659 / skip25 .235 (WORSE than 8B -- scale is not
monotone for skip on prose), q_int4 RTN .839 -> **GPTQ .9731**
(calibration is worth +0.134 at 32B vs +0.019 at 7B: calibration value
GROWS with scale). Leave-one-out: 60 layers, .913-.976, median .967
(flat: set-selection >> contiguous, placement law again).

## e2e (TP2, GPUs 0-1; nospec b1 prose 70.3, math 68.9)

| setting | ours | ratio | KnapSpec |
|---|---|---|---|
| b1/16k prose, greedy K4 | w4win-GPTQ 99.0 (acc 4.535) | 1.41x | 1.43x (tie) |
| b1 prose K5 / K6 | 91.2 / 93.4 (acc 5.13 / 6.00) | 1.30/1.33 | gamma* saturation exhibit |
| **b1/16k MATH (their task class), K5** | **w4win-GPTQ 112.0 (acc 5.361)** | **1.63x** | **1.43x — BEAT +14%** |
| b1 math K4 | 106.1 (acc 4.535) | 1.54x | also above |
| fp8-Marlin+win K5/K6 | 83.3/82.6 (acc 5.53/6.29) | 1.19/1.18 | bytes lose to W4 at b1 despite beta .96 |
| fp8-native+win K5 | 88.3 (acc 5.77) | 1.26 | kappa(M) at TP2, as fitted |
| T=0.7 w4win | 84.0 (acc 4.042) | 1.22 | greedy-only captured chain: sampled falls to eager floor (execution gap, named) |
| b8/16k w4win K5 | 529.6 (acc 5.200) | 1.28x | no counterpart |

## The two-scale verdict

- **8B (their weakest): 1.42x vs 1.28x — beat (+11%).**
- **32B (their flagship): 1.63x vs 1.43x task-matched — beat (+14%);
  1.41x tie on harder general prose.**
- Every winning margin is COMPOSITION: quant x window x floor-free
  chain. No single lever reaches any of these numbers (W4 alone 1.08x at
  8B-b1; fp8 variants <=1.26x at 32B; their lever prices <=1.16x on our
  measured frontiers at both scales).
- Their lever's own story sharpened: skip tolerance is family- AND
  scale-NON-monotone on general text (8B .754 -> 32B .659 contiguous),
  and the flat 32B leave-one-out (median .967) says their knapsack wins
  come from set-selection + task mix -- consistent with our placement
  law and our distribution study.
- Cross-system caveat: both sides report speedup vs their own AR
  baseline.
