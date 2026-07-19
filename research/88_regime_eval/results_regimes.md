# Phase 88 results — canonical regime benchmark (DRAFT)

> Serving driver, wall clock, 3 timed rounds (median; spreads <1%),
> Qwen3-8B 1xH100, fixed stack (wholechain + corrected skip-prefill +
> shared-KV), real datasets per README matrix. 2026-07-19.

## E1/E2: AR + current-best levers on every canonical regime

| regime | dataset (b, shape) | AR tok/s | W4+win K4 | W4+win K6 | W4A8-Hum K4 | W4A8-Hum K6 |
|---|---|---|---|---|---|---|
| R1 math CoT | GSM8K+AIME (b1, 1k out) | 149.3 | 1.18x (acc 4.79) | 1.19x (6.69) | 1.39x (4.80) | **1.42x** (6.71) |
| R2 conversation | MT-Bench (b1) | 149.7 | 1.04x (4.20) | 1.00x (5.38) | **1.29x** (4.41) | 1.25x (5.79) |
| R3 code | HumanEval (b8) | 822.5 | 1.14x (4.77) | 1.14x (6.51) | **1.34x** (4.75) | **1.34x** (6.54) |
| R4 summarization | CNN/DM 8k-doc (b8) | 591.2 | 0.84x (2.70) | 0.65x (2.90) | **0.94x** (2.71) | 0.76x (2.93) |
| R5 RAG QA | NQ-open+14k ctx (b8) | 356.4 | 1.29x (4.21) | 1.25x (5.39) | **1.42x** (4.44) | 1.41x (5.68) |
| R5cot RAG CoT | 14k ctx + AIME, 3k out (b8) | 585.7 | 1.51x (4.73) | 1.52x (6.50) | 1.73x (4.68) | **1.76x** (6.45) |
| R6 burst | GSM8K (b32, 256 out) | 2737.3 | 1.02x (4.74) | 0.94x (6.49) | **1.14x** (4.78) | 1.11x (6.53) |
| R7 translation | WMT14 de-en (b8) | 565.2 | 1.08x (4.50) | 0.97x (6.04) | **1.14x** (4.68) | 1.05x (6.30) |
| R8 RL rollout | MATH T=1.0 (b16, 2k out) | 2068.7 | 0.83x (3.92) | 0.72x (4.82) | **0.92x** (3.80) | 0.79x (4.81) |

(W4A8 arm note: first attempt wedged >70 min in flashinfer autotune at
warmup; identical retry booted in ~2 min and ran clean -- the
autotuner stall is NONDETERMINISTIC; watchdog + autotune-off fallback
added to the runner.)

K4-Humming UPDATE: the gap table shrinks to near-parity on cost alone
-- R4 0.84->0.94x, R8 0.83->0.92x (accept UNCHANGED ~2.7/3.8: the gain
is cheaper drafting + less rejected work at K4, not acceptance).
W4A8-Hum K4 is the per-regime winner at 6 of 9 regimes; the losing
cells' depth trend (K6 << K4) motivates the K2/K3 shallow probes
(queued). Window dig running for the accept side.

## Reading

- WINS (7 of 9 at per-regime best lever): math **1.42x**, conversation
  **1.25x**, code **1.34x**, RAG **1.41x**, RAG-CoT **1.76x**, burst
  **1.11x**, translation 1.08x. W4A8-Humming is the per-regime winner
  at 6 of the 7 wins -- the kernel-realization dividend generalizes
  across content types, not just the RAG headline cells.
- **R6 burst FLIPS to a win** (1.11x): the steady-state map prices
  b32/2k at 0.86x, but real short-answer bursts drain the batch into
  spec-favorable sizes AND the Humming kernel adds margin. The
  "adversarial batch regime" is beaten by lever choice on real data.
- LOSSES (the step-1 gap table, ordered):
  1. **R4 summarization, best 0.84x** (w4win K4; accept 2.70-2.93
     across all arms): pure accept starvation -- the faster W4A8
     kernel does NOT rescue it (0.76x at K6). Now measured on real
     CNN/DM instead of the artificial C4-continuation.
  2. **R8 RL rollout, best 0.83x** (accept ~3.9-4.8 at T=1.0):
     sampled-acceptance tax at b16. THE step-3 regime
     (EfficientRollout's home turf) -- must close before the
     DRAM-swap demo makes sense.
- K-selection matters regime-by-regime: K6 flips R7 win->loss on
  w4win (1.08->0.97) and deepens every loss; only R1/R5cot tolerate
  depth. Accept ~4.8/6 drafted at K6 in the losing cells means the
  marginal 2 draft positions are nearly free tokens thrown away.
- Cross-validation: R5cot reproduces both the 82-era w4win G1 cell
  (1.51 vs 1.53) and the T6 Humming record (1.76 vs 1.67 on 8-iter);
  R1/R2 AR ~149 matches the b1 record.

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
