# Phase 88 results — canonical regime benchmark (DRAFT)

> Serving driver, wall clock, 3 timed rounds (median; spreads <1%),
> Qwen3-8B 1xH100, fixed stack (wholechain + corrected skip-prefill +
> shared-KV), real datasets per README matrix. 2026-07-19.

## E1/E2: AR + current-best levers on every canonical regime

| regime | dataset (b, shape) | AR tok/s | W4+win K4 | W4+win K6 |
|---|---|---|---|---|
| R1 math CoT | GSM8K+AIME (b1, 1k out) | 149.3 | **1.18x** (acc 4.79) | 1.19x (6.69) |
| R2 conversation | MT-Bench (b1) | 149.7 | 1.04x (4.20) | 1.00x (5.38) |
| R3 code | HumanEval (b8) | 822.5 | **1.14x** (4.77) | 1.14x (6.51) |
| R4 summarization | CNN/DM 8k-doc (b8) | 591.2 | **0.84x** (2.70) | 0.65x (2.90) |
| R5 RAG QA | NQ-open+14k ctx (b8) | 356.4 | **1.29x** (4.21) | 1.25x (5.39) |
| R5cot RAG CoT | 14k ctx + AIME, 3k out (b8) | 585.7 | 1.51x (4.73) | **1.52x** (6.50) |
| R6 burst | GSM8K (b32, 256 out) | 2737.3 | 1.02x (4.74) | 0.94x (6.49) |
| R7 translation | WMT14 de-en (b8) | 565.2 | **1.08x** (4.50) | 0.97x (6.04) |
| R8 RL rollout | MATH T=1.0 (b16, 2k out) | 2068.7 | **0.83x** (3.92) | 0.72x (4.82) |

W4A8-Humming K6 arm: PENDING (first attempt wedged >70 min in
flashinfer autotune during warmup — the same stack booted in ~3 min
this morning; retry running with a 30-min cap + autotune-off fallback;
nondeterministic autotuner stall, flagged).

## Reading

- WINS (7 of 9 with per-regime best K): math 1.18x, conversation
  1.04x, code 1.14x, RAG 1.29x, RAG-CoT 1.52x, burst 1.02x,
  translation 1.08x. Cross-validation: R5cot reproduces the 82-era G1
  cell (1.51 vs 1.53) and R1/R2 AR ~149 matches the b1 record.
- LOSSES (the step-1 gap table, ordered):
  1. **R4 summarization 0.84x** (accept 2.70): the canonical
     low-accept regime, now on real CNN/DM instead of the artificial
     C4-continuation. Depth makes it worse (K6 0.65x) -- pure accept
     starvation. Gap to close: +19% just to reach parity.
  2. **R8 RL rollout 0.83x** (accept 3.92 at T=1.0 vs 4.74 greedy):
     sampled acceptance tax at b16. THE step-3 regime
     (EfficientRollout's home turf) -- must be fixed before the
     DRAM-swap demo makes sense.
- K-selection matters regime-by-regime: K6 flips R7 win->loss
  (1.08->0.97) and deepens every loss; K4 is the better static on
  canonical data. Only R1/R5cot tolerate K6.
- R6 burst NUANCE: steady-state map prices b32/2k at 0.86x, but the
  real workload measures 1.02x -- short answers drain the batch into
  spec-favorable sizes. Real bursts are milder than the steady cell;
  the trace/cell distinction matters for the paper's scoping claims.

## Step-1 dig directions (pre-registered)

- R4: window is likely the wrong draft-ctx lever for summarization
  (win512 beta was TASK-DEPENDENT: retrieval .890 vs ondist .975 --
  T5). Candidates: full-KV draft (no window) at b8, K2-3 shallow
  drafts (accept 2.7 supports gamma~2), kvq_fp8 draft-ctx instead of
  window (beta .974 on retrieval), OFF as floor via policy.
- R8: temperature-robust drafting -- KnapSpec-class T=0.7 gap was
  execution (greedy-only capture), not acceptance; verify sampled
  verify path; shallow K; accept-EMA policy would arm/disarm by
  request. EfficientRollout wins here with weight-quant self-spec at
  T=1.0 -- reproduce their operating point on OUR stack.
