# Detailed data tables — DRAFT (results not final; 2026-07-18)

> Every number is a committed measurement (phase artifact in research/).
> 8-iteration e2e runs unless noted; beta at 1152 paired positions.
> Figures: figures/figA_headtohead.png, figB_regimes.png,
> figC_frontiers.png, figD_serving.png, figE_policy.png,
> figF_rl_refresh.png.

## T1. Head-to-head vs KnapSpec (published) — all measured arms

### Qwen3-8B (their 1.28x)
| arm | b | ctx | K | tok/s | accept | speedup |
|---|---|---|---|---|---|---|
| AR baseline | 1 | 16k | - | 135.0±0.2 | - | 1.00 |
| W4 alone | 1 | 16k | 4 | 145.2±2.0 | 4.311 | 1.08 |
| **W4+win512 (fixed chain)** | 1 | 16k | 4 | **192.0±1.2** | 4.488 | **1.42** |
| W4+win512, T=0.7 | 1 | 16k | 4 | 177.8±1.2 | 4.378 | 1.32 (vs AR-T07 134.6) |
| W4 alone (short ctx 2k) | 1 | 2k | 4 | 183.8±3.5 | 4.477 | 1.23 (vs 148.9) |
| W4+win512 | 8 | 16k | 4 | 1124.7±2.6 | 4.506 | 1.81 (vs 620.6) |
| W4+win512 | 16 | 16k | 6 | 1503.7±15.8 | 5.689 | 1.79 (vs 838.3) |
| W4+win512 | 32 | 16k | 6 | 2001.7±17.9 | 5.776 | 2.34* (*AR capacity-capped) |

### Qwen3-32B, TP2 (their 1.43x)
| arm | prompts | K | tok/s | accept | speedup |
|---|---|---|---|---|---|
| AR baseline | prose | - | 70.3±0.6 | - | 1.00 |
| AR baseline | math | - | 68.9±0.1 | - | 1.00 |
| W4gptq+win512 | prose | 4 | 99.0±1.2 | 4.535 | 1.41 |
| W4gptq+win512 | prose | 5 | 91.2±0.6 | 5.128 | 1.30 (gamma* saturation) |
| W4gptq+win512 | prose | 6 | 93.4±0.5 | 6.000 | 1.33 |
| **W4gptq+win512** | **math** | **5** | **112.0±0.2** | **5.361** | **1.63** |
| W4gptq+win512 | math | 4 | 106.1±0.8 | 4.535 | 1.54 |
| fp8-native+win | prose | 5 | 88.3±0.3 | 5.765 | 1.26 (kappa(M) as fitted) |
| fp8-Marlin+win | prose | 5/6 | 83.3/82.6 | 5.53/6.29 | 1.19/1.18 (bytes lose to W4) |
| W4gptq+win, T=0.7 | prose | 4 | 84.0±0.1 | 4.042 | 1.22 (greedy-only capture gap) |
| W4gptq+win | prose b8 | 5 | 529.6±5.6 | 5.200 | 1.28 (vs 414.9) |
| W4A8-Humming+win | prose* | 4 | 57.5 wall | 4.33 | **~1.51** (serving-anchored) |
| **skip{2,4,7,16} x W4A8-Hum+win** | prose* | 4 | 57.5 wall | 4.33 | **~1.57 (+10% vs their 1.43)** |

*Humming rows measured on the serving driver (harness path blocked by
a Humming library bug at TP2 b1 -- odd-width lazy cubin-load hang,
filed); harness-equivalent = 1.41 x same-shape wall-ratio transfer
against the W4gptq+win anchor (anchor accept 4.46 matches the harness
arm's 4.535). The skip row is the TRIPLE composition: KnapSpec's own
lever (measured budget-4 greedy set, beta .892) x quantization x
windowed sparse attention, inside our framework.

## T2. Prior-phase e2e ledger (Qwen2.5-7B dense; MoE; MLA)

| model | cell | config | speedup | accept | provenance |
|---|---|---|---|---|---|
| Q2.5-7B | b32/16k | W4+win K6, fixed chain | **1.91** (=registered roofline) | 5.649 | 81-E3 |
| Q2.5-7B | b32/16k | same, K4 | 1.85 | 4.291 | 81-E3 |
| Q2.5-7B | b8/16k | same K4/K6 | 1.64/1.52 | 4.38/5.64 | 81-E3 |
| Q2.5-7B | b16/32k | same K4/K6 | **2.77±.48**/2.33±.32 | 4.33/5.62 | 79-e3x32 |
| Q3-30B MoE | b8/16k | win K3 fixed-stack | 1.15 | 3.766 | 81 MoE check |
| Q3-30B MoE | b4/2k | flr50 bf16-partial K2 | 1.03±.10 | 2.882 | 83-E2c (delivered flip) |
| Q3-30B MoE | b8,b32/2k | flr50 | 0.63/0.64 | 2.84/2.79 | 83-E2c (chain-blocked) |
| DS-V2-Lite | b32/16k | beta~1 self-draft K5 | 0.56 | 5.916 | 79 (chain-bound: OFF measured) |
| DS-V2-Lite | b32/16k | ngram K4 | 0.78 | 2.846 | 85-v6q (CPU-lookup psi=2.4) |

## T3. Beta singles at 16k (per model; 1152 paired positions)

| arm | Q2.5-7B | Q3-8B | Q3-32B | Q3-30B MoE | V2-Lite MLA |
|---|---|---|---|---|---|
| win512 | .978 | .975 | .971 | .971 | .922 |
| win128 | .976 | .975 | .961 | .969 | .576 |
| kvq_fp8 | .841 (no QK-norm) | **.985** | **.988** | .971 | .997 |
| q_fp8 (W-only) | .978 | .991 | .992 | .993 | 1.000 |
| q_int4 RTN | .924 | .952 | .839 | .972 | .997 |
| q_int4 GPTQ | .9427 | - | **.9731** (+.134!) | - | - |
| skip125 contig | .448 | .754 | .659 | .695 | .966 |
| skip25 contig | .09 | .537 | .235 | .40 | .32 |
| lr50 / flr50 | - | - | - | .826 / **.953** | .975 |

## T4. Skip-set greedy frontiers (budgets 1-7)

| model | 1 | 2 | 3 | 4 | 5 | 6 | 7 | contiguous ref |
|---|---|---|---|---|---|---|---|---|
| Q2.5-7B (28L) | .954 | .908 | .869 | .817 | .761 | .659 | .557 | .448 @12.5% |
| Q3-8B (36L) | .951 | .923 | .885 | .849 | .813 | .773 | .694 | .754 @12.5% |
| Q3-32B (64L) | .976 | .954 | .912 | .892 | .833 | .800 | .770 | .659 @12.5% |

## T5. New-lever gates (Phase 87, Qwen3-8B 16k) + retrieval workload axis

| arm | beta ondist | beta retrieval (needle@2k) | reading |
|---|---|---|---|
| actfp8 (A-only, dyn per-token e4m3) | **.9922** | - | A-quant factor isolated first time: ~free; validates W8A8 ~ W-only x A-only |
| w4a8 (int4-W + fp8-A) | .9470 | - | -.005 vs W4-alone (.952); e2e CONFIRMED dense-batch winner, see below |
| sparse24 (SparseGPT 2:4, 512 C4) | .8281 | - | calibration DOUBLES magnitude-2:4 (.416); still dominated alone (int4 .952 @ half the bytes) |
| s24w4 (2:4 + int4 = marlin_24 combo) | .8220 | - | int4 costs only -.006 on top of sparse; 0.125x bytes; kernel absent in fork -> alive-pending-realization |
| win512 | .975 | **.890** | window beta is TASK-DEPENDENT: first measured break on GQA |
| win128 | .975 | .872 | same collapse, deeper |
| kvq_fp8 | .985 | **.974** | holds; first regime where kvq beta > window beta (kvq pool still priced out; not built) |

W4A8+win e2e (Q3-8B 16k, GPTQ ckpt): Humming kernel **1.90x b8 / 2.19x
b16** (new batch headline) vs w4win 1.81/1.79; SAME ckpt on CutlassW4A8
1.60/1.78 -- realization span 1.60<->2.19 at identical beta (kernel choice
flips the cell). Accept 6.02 vs 5.69 (GPTQ-calibrated ckpt vs RTN proxy).

32B (TP2) W4A8 arm: b8/16k-prose K5 1.22x vs w4win 1.28x -- no flip
AT THAT CELL. SUPERSEDED BY THE CANONICAL SUITE (T8): on 9 regimes
the same two ckpts trade wins 5-2 (Humming takes b1/short-ctx/long-
decode; W4-GPTQ holds prefill-heavy b8 RAG and b32 burst) -- the
kernel-realization factor is BATCH/SHAPE-dependent at TP2, and
single-cell h2h understates lever diversity. b16/16k cell
capacity-infeasible at TP2 (KV pool 185k < 262k) -- 7th residency
incident, feasibility filter validated.

### T5b. Width-pruning frontier + kvq bound (Q3-8B, C4-profiled, 16k refs)

| arm | beta | ~byte cut | layer-skip frontier at same bytes |
|---|---|---|---|
| ffn125 | .9167 | 9.8% | .849 (+.07) |
| ffn25 | .8733 | 19.6% | .694 (**+.18**) |
| ffn375 | .8142 | 29.4% | - |
| ffn50 | .7613 | 39.1% | (2:4-SparseGPT .828 @ 50%, kernel-bound) |
| head25 / head50 | .8984 / .8038 | 4.3% / 8.7% | dominated by ffn per byte |
| ffn25head25 | .8273 | 23.9% | product .784 -> +.043 sub-additive |
| kvq_int4 | .9253 | KV 4x | fp8 .985; kvq lever bounded, still priced out |

On dense, CHANNEL granularity beats LAYER granularity at matched bytes
at every budget, with a free realization (smaller dense GEMMs).

w4+ffn compositions (measured betas .8993/.8628/.8116 at f=12.5/25/37.5%,
all sub-additive vs product by +.027-.037): PRICED OUT at every measured
8B cell (best 1.32x vs w4win 1.42x at b1; 1.58x vs W4A8-Humming 2.19x at
b16) -- dominates its class, wins no cell; e2e not built
(build-on-selection). Open: 32B b1, residency-constrained cells.

## T6. Serving wall-clock e2e (the deployment headline; 2026-07-19)

Full vLLM serving driver (LLM API), WALL time incl. prefill, real data:
C4 14k-token documents + AIME problem, 3072-token CoT outputs --
the realistic reasoning-serving shape (decode share ~90%). Qwen3-8B,
1xH100, fixed stack (whole-chain draft graph, corrected skip-prefill,
target-KV binding, mnb=8192).

| arm | b8 x 14k RAG | b16 x 14k RAG | aggregate |
|---|---|---|---|
| AR (no spec) | 581.6 tok/s | 769.9 | 694.9 |
| W4+win K4 | 1.53x | 1.72x | 1.64x |
| W4+win K6 | 1.51x | 1.76x | 1.65x |
| W4A8 K6 (CutlassW4A8) | 1.40x | 1.61x | 1.52x |
| **W4A8 K6 (Humming)** | **1.67x** | **1.90x** | **1.80x** |

- Implied b16 decode 1834 vs 862 tok/s = S_dec 2.13: the T5 harness
  record (2.19x) REPRODUCED in serving -- decode-cell maps transfer to
  wall once the serving stack is fixed and decode share is realistic.
- Kernel-realization span, third confirmation: the SAME W4A8 ckpt runs
  1.52x (Cutlass) vs 1.80x (Humming) wall aggregate.
- The serving-stack fixes this table required (each measured): draft
  prefill skip (2x prefill tax, with a placeholder-vs-prefill scheduler
  bug that hid ~0.3x wall in interim numbers), whole-chain draft graph
  (chain halved + boot-deterministic), async-scheduling force
  (draft_model silently disables it), mnb async-stall avoidance.

## T7. Runtime policy vs statics — three serving traces (fixed-skip
## stack, 2026-07-19)

Compiled policy (measure-on-deployment cells -> per-step scheduler
argmax w/ hysteresis + per-request accept EMA) vs every static arm,
full serving driver, Qwen3-8B. Aggregate tok/s per trace
(82/results_e2.md, valfx_*.json):

| trace (regime mix) | OFF | K4 | K6 | policy |
|---|---|---|---|---|
| mixed (b1 AIME / b8 docs / b32 AIME) | 955.7 | 976.6 | 914.0 | **979.2 (wins)** |
| long-math RAG (b1/8/16 rag14k + b32 aime) | 584.4 | **769.8** | 753.5 | 752.2 (−2.3%) |
| long-ctx docs (b1/8/16 doc16k + b32 aime) | 768.1 | 799.5 | 704.9 | **817.8 (wins)** |

- One policy config vs three DIFFERENT per-trace static winners (K4
  wins long-math, OFF near-wins mixed, no static wins long-ctx): the
  policy wins 2 of 3 outright, −2.3% on the third. Claim = regret
  bound (1–3% of per-trace oracle) + occasional outright wins.
- Compiled cells (fixed skip) span 1.59x (b16/14k K4) to 0.83x
  (b16/2k): regime deltas within ONE column. Scoping law: intra-column
  switching ceiling +10.4% max over all workload mixes (predicted from
  these cells); cross-column fleet ceiling +39.1% — selection across
  columns is the effectiveness headline, runtime tracking is the
  regret-bound story.
- Skip-bug provenance: pre-fix versions of this table (archived
  *_buggyskip) throttled spec-win cells ~20% and accidentally
  protected spec-losing cells (alternating K=0 = 50% duty cycle) —
  both arms and policy re-measured on the corrected stack.

## T8. Canonical regime suite (Phase 88): the framework finds an
## AR-beating setting at 9/9 regimes (8B); winners split at 32B

Real datasets per regime (Spec-Bench/KnapSpec/EfficientRollout-
grounded: GSM8K+AIME, MT-Bench, HumanEval, CNN/DM, NQ-open+ctx,
WMT14, MATH@T=1.0), serving wall, 2026-07-19/20.

| regime | 8B best (all W4A8-Hum) | 32B best |
|---|---|---|
| math CoT b1 | K6 **1.42x** | skip x Hum K4 **1.50x** |
| conversation b1 | K4 1.29x | Hum K5 1.36x |
| code b8 | K4 1.34x | Hum K5 1.39x |
| summarization b8 | K2 **1.08x** (was 0.84 loss) | Hum K3 1.00x |
| RAG QA b8/14k | K4 1.42x | **W4-GPTQ K4** 1.13x |
| RAG CoT b8/14k | K6 **1.76x** | skip x Hum K4 **1.52x** |
| burst b32 | K4 1.14x | **W4-GPTQ K5** 1.19x |
| translation b8 | K4 1.14x | Hum K5 1.06x |
| RL rollout b16 T=1.0 | K2 **1.05x** (was 0.83 loss) | OFF (best spec 0.95x)* |

*32B R8 blocked by the Humming odd-width bug (8B evidence predicts a
win). Key findings: acceptance is FRONT-LOADED in depth (shallow-K
converts the losing regimes); at 8B one lever dominates and selection
= depth; at 32B winners split across kernel x depth x composition x
OFF. Selection accounting: oracle composite 1.26x (8B) / 1.20x (32B)
vs AR; switching bound over best static +4.2% / +7.3% (uniform mix),
CONCENTRATED where statics lose outright (per-cell +7-33%).

## T9. RL-rollout staleness + DRAM lever refresh (Phase 89, 8B)

Drift trace (5 phases, calibrated weight-drift, b16 x MATH T=1.0,
drain shape), full serving driver. Swap mechanics: 6.07GB draft
pinned-DRAM->GPU **113 ms**; CUDA-graph replay after in-place copy_
is bit-exact (zero-downtime, no re-capture); live mid-serving swap
measured 114-116 ms.

| arm | aggregate vs AR |
|---|---|
| stale draft, static (never refreshed) | 0.55-0.76x |
| per-step runtime policy alone | 1.00x |
| **full system: policy + detector-fired DRAM refresh** | **1.055x** |

The staleness cliff (-45%) is converted to a bound around the
per-cell fresh optimum; the detector (accept EMA at 10s cadence,
S<1-boundary gate) fires the refresh mid-serving and probes re-arm
within the phase. Measured anti-patterns retained: tight polling
(-5%), eager gates (0.89x), disk-loading in the swap path (~1s/fire).
The demo cell is the map's THINNEST (fresh ceiling 1.05-1.07x); the
same protection transfers to the 1.4-1.9x cells.

## Notes / flags
- DRAFT: iteration counts modest (4-8); b32 8B AR capacity-capped; T=0.7
  rows carry the greedy-only-capture execution gap; cross-system caveat
  (speedup vs own AR both sides); KnapSpec numbers from their paper.
- T1-T5 are decode-focused cell measurements (harness); T6 is the
  serving WALL headline; T7 is the runtime-selection record; T8 is
  the canonical-benchmark coverage claim; T9 is the RL/adaptation
  system. Cells feed the map, T6/T8 are what a deployment sees,
  T7/T9 are the selector and swap machinery operating live.
- Composition status (measured): skip x quant x window stacks at 32B
  (+3-6% on either kernel, accept cost ~0 -- three sub-additivity
  confirmations); skip priced out at 8B (scale-keyed). kvq draft-ctx
  flagged as the accept-side headroom for summarization (window
  accept is a step function of draft ctx; full-ctx priced out at
  R~1.0 attention-bound).
