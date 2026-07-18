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

## Entry summary

| lever | status |
|---|---|
| activation quant (fp8-A; W4A8) | IN POOL; **e2e-confirmed dense-batch winner** (Humming realization: 1.90x b8 / 2.19x b16 vs w4win 1.81/1.79) |
| SparseGPT 2:4 (+int4 combo) | alive-pending-realization (kernel absent) |
| draft-only kvq pool | not selected (regime found but priced out); reference column upgraded with the retrieval row |
