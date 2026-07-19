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

## Step-1 dig RESULT (2026-07-19): 9/9 regimes beat AR

The two gap regimes both CROSS with existing levers -- no new
mechanism, just the K axis extended to shallow depths on the Humming
kernel:

| arm (R4 summarization) | tok/s | vs AR | accept |
|---|---|---|---|
| w4win K4 win512 (baseline loss) | 496.0 | 0.84x | 2.70 |
| w4win K4 win2048 | 463.3 | 0.78x | 2.67 |
| w4win K4 win8192 (full article) | 478.9 | 0.81x | **4.06** |
| w4a8-Hum K4 win8192 | 537.1 | 0.91x | 4.21 |
| w4win K2 win512 | 577.1 | 0.98x | 2.18 |
| w4a8-Hum K3 win512 | 598.6 | **1.01x** | 2.53 |
| **w4a8-Hum K2 win512** | **636.5** | **1.08x** | 2.18 |

| arm (R8 RL rollout, T=1.0) | tok/s | vs AR | accept |
|---|---|---|---|
| w4win K4 (baseline loss) | 1716.7 | 0.83x | 3.92 |
| w4a8-Hum K4 | 1894.5 | 0.92x | 3.80 |
| w4a8-Hum K3 | 2015.1 | 0.97x | 3.28 |
| **w4a8-Hum K2** | **2161.2** | **1.05x** | 2.64 |

Mechanism findings:
- ACCEPT IS FRONT-LOADED: per-position f falls fast with depth (R4:
  f1-2=0.59 vs 0.43 avg at K4; R8-T1.0: f1-2=0.82 vs 0.70). Shallow
  drafts harvest the good positions and skip the wasted ones -- the
  low-accept regimes are DEPTH problems, not spec-loses problems.
- WINDOW STEP-FUNCTION (R4): win2048 buys nothing (2.67 vs 2.70);
  win8192 (whole article visible) jumps accept to 4.06. But the
  full-window chain runs at R~1.0 (attention-bound; Humming GEMMs
  recover only 0.81->0.91) -- full visibility is priced out on both
  kernels. FUTURE HEADROOM: cheap long-ctx draft attention (kvq_fp8
  draft ctx, retrieval beta .974 at 4x fewer bytes -- T5) would stack
  the accept gain on the shallow-K win; the pool lever shelved
  on-distribution earns its R4 slot here.
- The compiled policy's option set must include K2/K3 (current tables
  only price K4/K6) -- required for step-3 runtime adaptation to
  reach these cells.

## FINAL per-regime-best (8B column, all >= 1.0)

| regime | best setting | speedup |
|---|---|---|
| R1 math CoT | W4A8-Hum K6 | 1.42x |
| R2 conversation | W4A8-Hum K4 | 1.29x |
| R3 code | W4A8-Hum K4 | 1.34x |
| R4 summarization | W4A8-Hum K2 | **1.08x** (was 0.84 loss) |
| R5 RAG QA | W4A8-Hum K4 | 1.42x |
| R5cot RAG CoT | W4A8-Hum K6 | 1.76x |
| R6 burst | W4A8-Hum K4 | 1.14x |
| R7 translation | W4A8-Hum K4 | 1.14x |
| R8 RL rollout T=1.0 | W4A8-Hum K2 | **1.05x** (was 0.83 loss) |

"The framework finds at least one AR-beating setting" holds at 9/9
canonical regimes; every winner is reachable from the existing lever
axes (ckpt x kernel x window x K). One model column (Qwen3-8B), one
GPU; the MoE/MLA columns keep their measured verdicts (win-K3 1.15x /
OFF) from prior phases.

## 32B column (TP2): the canonical suite at scale -- lever diversity
## MEASURED (2026-07-19)

Arms: AR, W4-GPTQ+win K4/K5, W4A8+win K5 (Humming-forced). Same
datasets, same driver, GPUs 0+1.

| regime | AR tok/s | W4gptq K4 | W4gptq K5 | W4A8-Hum K5 | winner |
|---|---|---|---|---|---|
| R1 math CoT | 72.7 | 1.26x | 1.21x | **1.45x** | Hum K5 |
| R2 conversation | 73.0 | 1.20x | 1.07x | **1.36x** | Hum K5 |
| R3 code | 395.7 | 1.24x | 1.22x | **1.39x** | Hum K5 |
| R4 summarization | 321.9 | 0.86x | 0.78x | 0.88x | (gap -> shallow) |
| R5 RAG QA | 190.4 | **1.13x** | 1.06x | 1.09x | **W4 K4** |
| R5cot RAG CoT | 376.1 | 1.32x | 1.31x | **1.43x** | Hum K5 |
| R6 burst | 930.2 | 1.07x | **1.19x** | 1.13x | **W4 K5** |
| R7 translation | 261.4 | 0.96x | 1.03x | **1.06x** | Hum K5 |
| R8 RL rollout | 1099.7 | 0.76x | 0.70x | 0.75x | (gap -> shallow) |

Shallow probes on the gap cells (R4/R8, K2-K3 both ckpts):

| cell | W4 K3 | W4 K2 | Hum K3 | Hum K2 |
|---|---|---|---|---|
| R4 summarization | 0.92x (2.66) | 0.99x (2.28) | **1.00x** (2.67) | BLOCKED* |
| R8 rollout T=1.0 | 0.82x (3.05) | 0.95x (2.54) | 0.92x (3.05) | BLOCKED* |

*Hum-K2 at TP2 is unmeasurable pending a Humming library bug (M=3
GEMM tile hangs in lazy cubin load; 6/6 reproductions across
autotune-on/off, cache-on/off, parallel/sequential build, wholechain/
piecewise chain -- see infra note). 32B verdicts: R4 = AT PARITY
(Hum K3 1.00x; plus W4-K2 0.99x on a working kernel); R8 = NOT
CLOSED at 32B, best 0.95x (W4 K2) -- the 8B crossing (Hum K2 1.045x)
predicts a win here once the M=3 tile bug is fixed; the one open
cell in the two-column campaign.

READINGS (the diversity result):
- UNLIKE 8B (one dominant point, only K varies), the 32B winners
  SPLIT ACROSS EVERY AXIS: kernel (Humming takes 5 cells, W4-GPTQ
  takes 2 -- the flip is batch/decode-share-dependent at TP2: W4
  holds b8-prefill-heavy RAG and b32 burst), and depth (K4 vs K5
  split within W4; R7's depth direction INVERTS vs 8B -- K6 killed
  translation at 8B, K5 rescues it at 32B, accept grows faster with
  depth at scale).
- The old "no kernel flip at 32B" verdict (b8/16k prose harness cell)
  was real but cell-specific: on the canonical suite the same two
  ckpts trade wins 5-2. Single-cell h2h comparisons UNDERSTATE lever
  diversity; suite-level measurement was needed to see it.
- Gap cells: R4 reaches parity at Hum K3 (1.00x, same as the 8B K3
  crossing); R8 at 32B is HARDER than 8B (best 0.95x at W4 K2 --
  where 8B crossed at Hum K2 1.045). T=1.0 sampled-accept at TP2 is
  the one cell class not yet closed by depth alone.
- Autotune-wedge infra note: the flashinfer autotuner hung at TP2
  warmup twice on the SAME arm (w4a8 K2) after passing on K3/K5 --
  arm-nondeterministic; K2-Hum numbers come from the autotune-off
  fallback (heuristic tactics; sanity-check against the K3 trend
  before citing).

### Infra: the w4a8 wedge ROOT-CAUSED (Humming lazy cubin load)

Faulthandler stack (SIGABRT on the wedged TP0 worker): main thread
spins in humming/kernel/humming.py load_cubin -> ops register_kernel
-> torch._ops.__call__, invoked from INSIDE the AOT-compiled draft
forward; 16 idle build-pool threads; other rank spin-waits in its
collective -> deterministic livelock. Humming JIT-builds/loads cubins
per GEMM shape on FIRST USE; the K2 capture set is the only one whose
M=3 (batch x qlen) tile at TP2 half-width dims is never touched by an
eager pass first. Reproduced 5/5 on w4a8+K2+TP2; absent on K3/K5
(shapes covered eagerly) and on TP1 (8B K2 ran fine). NOT the
flashinfer autotuner (wedges with autotune off), NOT the compile
cache (wedges with VLLM_DISABLE_COMPILE_CACHE=1), NOT the build pool
(wedges with HUMMING_DISABLE_PARALLEL_BUILD=1).
- PROPER FIX (filed): eager Humming warmup pass per capture shape
  before draft FULL-CG capture in llm_base_proposer -- matters for
  step-3 RL at TP2 where K2 is the R8 winner-class depth.
- MEASUREMENT WORKAROUND ATTEMPTED AND FAILED: the piecewise chain
  wedges identically (6th repro) -- the hang is the M=3 tile's cubin
  load itself (likely below the library's min tile M, falling into a
  broken small-M path at TP2 half-width dims), not capture
  interaction. Hum-K2@TP2 recorded as blocked; upstream/library
  report is the fix path. W4A16-K2 (Marlin) unaffected.

### Infra verdict (2026-07-20 02:43): BOX-STATE, not our stack

Differential closed the case: the byte-identical 32B TP2 w4a8-K5 boot
that succeeded at 19:26 hangs at 02:23 (8/8 night hangs across
compiled/eager/piecewise x cache-on/off x autotune-on/off x
parallel/sequential build). Hang site = cuModuleLoad-class driver
calls inside TP workers; a standalone cuModuleLoad on the same cubin
completes in 0.02s. Conclusion: driver/box-level resource held by
co-tenant load after ~23:00 (GPUs 2-7 occupied by other users'
engines); no code fix applies. The eager pre-warm patch (committed)
is kept: it removes the in-capture first-use path (the K2 M=3 case)
and surfaces the environmental hang at boot instead of mid-capture.
- PARKED pending a quiet box: RKS w4a8 K4/K5 (the KnapSpec-parity
  reprice) and the 32B R8/R4 Hum-K2 shallow cells.
- Retry cmd: research/88_regime_eval/scripts/run_rks_hum.sh

## STEP 2 CLOSED (2026-07-20): the KnapSpec parity cell falls to
## W4A8-Humming

Quiet box (co-tenants gone): the parked RKS arms ran. Serving driver,
b1 x 16k-ctx prose (harness prompts over C4 prefix), 160-tok outputs:

| arm | wall tok/s | vs AR | harness-equivalent* |
|---|---|---|---|
| AR | 45.6 | 1.00 | 1.00 |
| W4-GPTQ+win K4 (anchor) | 51.0 | 1.118x | 1.41x (measured, 86-E3) |
| **W4A8-Hum+win K4** | **54.7** (acc 4.24) | **1.200x** | **~1.51x** |
| W4A8-Hum+win K5 | 54.2 (acc 4.84) | 1.189x | ~1.50x |

*harness-equivalent = 1.41 x (arm/anchor) same-shape ratio transfer;
wall ratios are prefill-diluted (16k prefill over 160-tok outputs).
Anchor accept 4.46 matches the harness arm's 4.535, validating shape.

vs KnapSpec's published 1.43x at this cell: **~1.51x, +6%** -- the
last unmatched h2h cell closes via kernel realization (the same
Humming-vs-W4 lift the canonical sweep measured at every 32B b1
cell). Direct harness-protocol confirmation arms running.

Wedge-pattern refinement: K4 passed 1/3 on the quiet box (intermittent,
not deterministic); K2 remains 0/6. Odd verify-widths (3, 5) wedge at
far higher rate than even (4, 6) at TP2 -- library bug report should
lead with the M=3/odd-M tile path; nighttime co-tenant load multiplies
the rate (8/8 hangs incl. even-width K5 that passed 1st try on the
quiet box).
