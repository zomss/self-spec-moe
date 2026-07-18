# Phase 87 results — three levers gated (DRAFT)

## L2 Activation quant — ENTERS THE POOL (kernel-ready)

| arm | beta (Q3-8B, 16k) | reading |
|---|---|---|
| actfp8 (A-only, dynamic per-token e4m3) | **0.9922** | the A-quant factor isolated for the first time: essentially free; validates W8A8 ~ W-only x A-only decomposition |
| w4a8 (int4-W + fp8-A) | **0.9470** | -0.005 vs W4-alone (.952): the CutlassW4A8 config is beta-viable |

Regime: batch rows (compute-bound verify/high-M draft); b1 stays
weight-only (measured at 32B).

### W4A8 e2e (Qwen3-8B, 16k, win512+fixed chain; GPTQ ckpt, pack-quantized)

Ckpt gotcha: llm-compressor saves W4A8 as int-quantized (plain int
`weight`); vLLM's CT-W4A8 schemes load only pack-quantized
(`weight_packed`) -> force `quantization_format="pack-quantized"` on save.

| arm | b8 K4 (AR 620.6) | b16 K6 (AR 838.3) |
|---|---|---|
| w4win (W4A16 Marlin, ref) | 1124.7 = 1.81x (acc 4.51) | 1503.7 = 1.79x (acc 5.69) |
| w4a8win / CutlassW4A8 | 993.0 = **1.60x** (acc 4.54) | 1489.0 = 1.78x (acc 5.94) |
| w4a8win / HummingW4A8 | 1181.3 = **1.90x** (acc 4.58) | 1838.8 = **2.19x** (acc 6.02; repro 1808.1 = 2.16x) |

- **W4A8-Humming is the new dense-batch winner** (b8 +0.09x, b16 +0.40x
  over w4win) -- the priced batch flip confirmed e2e.
- **Realization span 1.60x <-> 2.19x at identical beta** (same ckpt, two
  W4A8 kernels): the strongest single instance of the realization-pricing
  thesis yet. Kernel choice = the difference between losing and winning
  the cell.
- Accept HIGHER than w4win (6.02 vs 5.69 at K6): the e2e ckpt is
  GPTQ-calibrated while the offline w4a8 gate (.947) used the RTN proxy;
  the +~.02 beta matches the measured GPTQ calibration gain at 7B (+.019).
- Cutlass was silently disabled by the harness's Marlin-forcing
  VLLM_DISABLED_KERNELS default -> that accident surfaced Humming; both
  kernels then measured deliberately.

## L1 Calibrated 2:4 pruning — calibration DOUBLES it; still dominated alone; combo marginal

| arm | beta | vs |
|---|---|---|
| sparse24 (SparseGPT 2:4, 512 C4) | 0.8281 | magnitude 2:4 was 0.416 -- the fair-form demand vindicated (mirrors GPTQ +0.134 at 32B) |
| s24w4 (2:4 + RTN int4 = marlin_24 combo) | 0.8220 | int4 costs only -0.006 ON TOP of sparse (error overlap) |

- 2:4-alone remains DOMINATED: 0.828 at 50% bytes vs int4 0.952 at 25%.
- The combo (0.125x bytes) prices ~level with w4win at 8B/b1 (1.35 vs
  1.31 rough) -- marginal, and the fork LACKS the marlin_24 kernel (E0-C2).
- Verdict: **gated-alive-pending-realization** (wide-sigma priced; kernel
  port decision deferred; a SparseGPT+GPTQ JOINT recipe may beat the
  stacked-RTN combo beta and is the fair form of the combo if pursued).

## L3 Retrieval workload axis + kvq pool — REGIME FOUND, pool still not selected

Needle bank (fact at depth 2k, outside any window; Q3-8B, 12 prompts):

| arm | beta ondist | beta retrieval | delta |
|---|---|---|---|
| win512 | .975 | **.890** | -0.085 -- window beta is TASK-DEPENDENT (first measured break on GQA) |
| win128 | .975 | .872 | -0.103 |
| kvq_fp8 | .985 | **.974** | holds; FIRST regime where kvq beta > window beta |

- Map consequence: retrieval DISCOUNTS window configs ~15-20% (composed
  w4win at b32/16k reprices ~1.9 -> ~1.6) but kvq still loses the cell
  (2x byte cut can't beat 30x even with the beta lead) -> **the kvq pool
  is NOT built** (the build-on-selection discipline holds).
- The WORKLOAD AXIS enters the map: window viability now carries a
  task feature (long-range dependence), alongside the ngram lesson
  (output repetitiveness). Two measured workload features total.

## L4 (follow-up gates) — kvq bounded; WIDTH PRUNING BEATS LAYER SKIP

Requested completeness pass: kvq full-potential form + the untried
width-pruning axis (Wanda-style importance, 32 C4 seqs -- disjoint from
eval refs; masks are exact removal equivalents; realization is FREE:
smaller dense GEMMs, no special kernel).

| arm | beta | ~total-byte cut | matched-byte comparator |
|---|---|---|---|
| kvq_int4 (per-head/token, g=head_dim) | .9253 | KV 4x (vs fp8 2x) | kvq_fp8 .985: int4 costs -.06; still priced out vs window -> kvq lever now BOUNDED top-to-bottom |
| ffn125 (FFN-channel 12.5%) | .9167 | 9.8% | greedy layer-skip 4L=11.1%: .849 (+.07) |
| ffn25 | .8733 | 19.6% | greedy 7L=19.4%: .694 (**+.18**) |
| ffn375 | .8142 | 29.4% | (skip frontier ends ~.69 @ 19%) |
| ffn50 | .7613 | 39.1% | SparseGPT 2:4 .828 @ ~50% -- 2:4 wins deep, ffn wins realization (no kernel needed) |
| head25 (Q-heads) | .8984 | 4.3% | poor per-byte (Q/O only 17% of layer bytes) |
| head50 | .8038 | 8.7% | dominated by ffn125 (.917 @ 9.8%) |
| ffn25head25 | .8273 | 23.9% | product predicts .784 -> **+.043 sub-additive** (error overlap, like s24w4) |

- **FFN-channel pruning ENTERS THE POOL**: it beats the layer-skip
  frontier at every matched byte budget on dense (+.07 shallow, +.18
  deep) and needs no kernel. The placement law refines: on dense,
  CHANNEL granularity > LAYER granularity at matched bytes.
- Head pruning: measured-dominated (per-byte loser to ffn at every
  budget).
- Width x width combo is sub-additive (favorable) -- third measured
  product-law exception, same sign as s24w4.
### w4+ffn composition: measured beta, PRICED OUT at 8B (no e2e built)

Combo betas MEASURED (sub-additive at every depth): w4ffn125 .8993
(product .8727), w4ffn25 .8628 (.8314), w4ffn375 .8116 (.775).
Priced on measured Q3-8B anchors (e2e-implied R, term edit
1 - s*0.783*f with weight-stream share s = .70/.50/.45 +- sigma at
b1/b8/b16; 2000 MC, LCB05 -- scripts/price_w4ffn.py):

| cell | winner (P) | best w4+ffn (P) |
|---|---|---|
| b1/16k | w4win 1.42x (.89) | w4ffn25win 1.32x (.04) |
| b8/16k | w4a8win-Humming 1.90x (.95) | 1.57x (.00) |
| b16/16k | w4a8win-Humming 2.19x (1.00) | 1.58x (.00) |

- The -0.09 beta vs w4win buys only ~10-15% R at 8B (draft not
  weight-bound enough) -> **no cell flips; e2e NOT built**
  (build-on-selection).
- Pool reading: ffn-width DOMINATES ITS CLASS (beats layer-skip at all
  matched bytes) yet wins no cross-class cell -- the
  whole-combination-search argument in one lever.
- Open cells where it could flip (map-nominated, unmeasured): 32B b1
  (more weight-bound), residency-infeasible cells (byte cut buys
  feasibility).

## L2 continued — 32B W4A8 arm (map v6.3 standing queue item): NO FLIP

Ckpt: Qwen3-32B-W4A8-gptq (512 C4, pack-quantized). TP2, GPUs 0-1, 16k.

| arm (b8 K5) | tok/s | speedup | accept |
|---|---|---|---|
| nospec | 414.9 | 1.00 | - |
| w4win K5 (incumbent) | 529.6 | **1.28x** | 5.20 |
| w4a8win Humming K5 | 505.9 +-42 | 1.22x | 5.16 |

- **The 8B batch flip does NOT transfer to 32B b8**: implied
  Humming/Marlin kernel factor ~1.05 here vs 0.945 at 8B b8 -- the
  realization factor is SCALE/TP-DEPENDENT (TP2 halves GEMM shards;
  dynamic act-quant overhead stops amortizing).
- Map vindicated: v6.3's LCB conservatism kept the W4A8 transfer row
  from winning any Q2.5 cell; the measured confirm agrees (incumbent
  holds).
- b16/16k (the 8B-evidence flip cell): **CAPACITY-INFEASIBLE at TP2**
  -- engine KV pool 185,360 tok < 262,144 needed; the w4win "0.58x"
  print is KV-thrashing, not a measurement. 7th validated residency
  incident; cell marked infeasible (TP4 would be a different hardware
  column).

## Entry summary

| lever | status |
|---|---|
| activation quant (fp8-A; W4A8) | IN POOL; **e2e-confirmed dense-batch winner** (Humming realization: 1.90x b8 / 2.19x b16 vs w4win 1.81/1.79) |
| SparseGPT 2:4 (+int4 combo) | alive-pending-realization (kernel absent) |
| draft-only kvq pool | not selected (regime found but priced out); lever bounded: fp8 .985 / int4 .925 |
| FFN-channel width prune | **IN POOL** (beats layer-skip frontier at all matched bytes; realization-free) |
| Q-head prune | measured-dominated by ffn at every budget |
