# Results: Quantized Experts -- Lossy Serve-FP8 vs Lossless FP8-Draft

Date: 2026-06-24

Real vLLM single-GPU decode (Qwen3-30B-A3B, dummy weights, TP=1) + a fake-quant
acceptance study. Corrects an earlier framing error: quantizing the *whole* model
is lossy; the lossless use of quantization is an FP8 *draft* with a bf16 *verify*.

## 1. Lossy: serve FP8 instead of bf16

| quantization | B=64 | B=256 | gain |
| --- | ---: | ---: | ---: |
| bf16 | 24.98 | 30.71 | 1.0x |
| FP8 (W8A8) | 7.31 | 13.95 | **3.42x / 2.20x** |

This is ~2.2-3.4x throughput, but it is **lossy** -- the FP8 model is a different,
lower-quality model. Valid only if FP8 output quality is acceptable. Not
speculation, not lossless. (bf16 matches Phase 16, harness validated.)

## 1b. FP8 gain trend across batch (ms/token, single GPU)

| batch | bf16 | FP8 | gain |
| ---: | ---: | ---: | ---: |
| 1 | 5.28 | 5.12 | 1.03x |
| 4 | 9.23 | 6.10 | 1.51x |
| 16 | 17.00 | 5.85 | 2.91x |
| 64 | 24.96 | 7.34 | **3.40x** |
| 256 | 30.69 | 14.01 | 2.19x |
| 1024 | 69.05 | 78.98 | 0.87x |

The FP8 gain is **regime-dependent**, peaking at moderate batch:
- **B=1 (~1.0x):** only `top_k` experts are activated, so the expert-weight read is
  tiny -> nothing for FP8 to save. Quantization barely helps at the pure-latency point.
- **B=16-64 (~3x):** all experts are activated and the step is memory-bound on
  expert weights -> FP8 halves the dominant cost. FP8 ms/token stays nearly flat
  (5.1->7.3) while bf16 climbs steeply (5.3->25.0): FP8 pushes the memory wall out.
- **B>=256:** decode goes compute-bound; the gain falls. **B=1024 inverts (0.87x) --
  a kernel artifact** (default, non-autotuned FP8 Triton MoE config; a tuned FP8
  kernel should stay >= bf16). Do not read the inversion as fundamental.

Implication: FP8 (and the FP8-draft scheme) is a **moderate-batch throughput** lever,
not a low-batch latency one -- at B=1 there is too little expert read to quantize.
This is a different regime from the inter-node *latency* win that local routing
targets, reinforcing that the two are complementary, not competing.

## 2. Is FP8 a good *draft* of the bf16 model? (acceptance vs bf16 target)

| draft | sampled acc | top1 |
| --- | ---: | ---: |
| bf16 + local (M=E/2) | 0.825 | 0.750 |
| **FP8 + full routing** | **0.954** | 0.875 |
| FP8 + local | 0.823 | 0.812 |

**FP8 rounding is nearly invisible to the next-token distribution (0.954 acceptance
vs the bf16 target).** Combining FP8 with local routing is pointless: it falls back
to ~0.82 (local error dominates) for no gain.

## 3. Local routing adds nothing under FP8 (timing)

| draft | B=64 | B=256 |
| --- | ---: | ---: |
| FP8 + full (E=128) | 7.31 | 13.95 |
| FP8 + local (E=64) | 7.53 | 14.71 |

Once quantized, the expert weights are already small, so reducing the expert count
gives **no speedup** (E=64 ~= E=128). Local routing's lever vanishes under FP8 and
only costs acceptance -- so **do not combine FP8 with local routing**.

## 4. Lossless: FP8 draft + bf16 verify (the right scheme)

For a lossless method the verify must be bf16. Use FP8 only for the draft:
- **Draft:** FP8, full routing -> 0.29x a bf16 step (7.31 vs 24.98), acc 0.954.
- **Verify:** bf16, full routing -> exact.

Because the FP8 draft is both cheap and accurate, long drafts pay off:

| k | throughput gain (lossless) |
| ---: | ---: |
| 2 | 1.65x |
| 4 | 1.84x |
| 6 | **1.90x** |

**~1.9x LOSSLESS** (exact bf16 output) -- it captures most of the lossy serve-FP8
2.2x, because at 0.95 acceptance the expensive bf16 verify runs ~once per cycle and
cheap FP8 drafts do the rest. This is the "cheap draft enables long k" lever that
**layer-skip could not deliver** (Phase 17): FP8 error is tiny (0.954) where 1/8
layer-skip already fell to 0.73.

## 5. The cost (per the lossless-verify constraint)

The verify needs bf16 experts and the draft needs FP8 experts, so you store both
~= **1.5x expert memory**. Runtime downcast does not help (reading bf16 to make FP8
saves nothing on the draft read), so the FP8 copy must be resident. 1.5x memory
buys ~1.9x lossless throughput.

## Summary

| approach | gain | lossless? | notes |
| --- | ---: | --- | --- |
| Serve FP8 (whole model) | ~2.2x | no | different model |
| **FP8-draft + bf16-verify self-spec** | **~1.9x** | **yes** | 1.5x expert memory |
| bf16 local-draft self-spec | ~1.07x | yes | Phase 16 |
| layer-skip + local | <1.0x | yes | Phase 17, negative |

## Conclusion

Corrected: the earlier "FP8 2.2x composes with local-draft for 2.35x" was a category
error -- the 2.2x is lossy, and FP8 does not compose *with* local routing (it
replaces it). The honest result is stronger and cleaner: an **FP8 draft of the same
model, verified in bf16, is a ~1.9x LOSSLESS throughput win** -- the cheap-and-
accurate draft layer-skip never was. Local routing is unnecessary once quantized.
This is an intra-node, communication-free, lossless gain; the genuine novelty of
local-expert self-speculation remains the inter-node *latency* angle, while for
throughput the answer is simply "use an FP8 draft."
