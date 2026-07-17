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
