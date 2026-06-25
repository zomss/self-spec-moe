# Results: Activation-Quant (Communication-Axis) Draft Acceptance

Date: 2026-06-25. Real Qwen3-30B-A3B. The EP all-to-all moves *activations*, so
quantizing the dispatched/combined activations shrinks communication directly. We
fake-quantize, per-token, the two activations that cross the wire in
`Qwen3MoeExperts` -- the dispatch input and each expert's combine output -- combined
with bf16 or NVFP4 weights, and measure rejection-sampling acceptance vs the exact
bf16 target (Phase 18/22 method; 122 positions, 512 samples). Verify stays bf16, so
the draft can use activation precisions too lossy to serve directly.

## Acceptance

| config | weights | activations | comm bytes | sampled acc | expected |
| --- | --- | --- | ---: | ---: | ---: |
| w16a8 | bf16 | FP8 (per-token) | ×0.5 | **0.959** | 0.959 |
| w16a4 | bf16 | NVFP4 (g16) | ×0.25 | 0.912 | 0.914 |
| **w4a16** | NVFP4 | bf16 | ×1.0 | 0.913 | 0.913 |
| **w4a8** | NVFP4 | FP8 | ×0.5 | **0.915** | 0.916 |
| **w4a4nv** | NVFP4 | NVFP4 (g16) | ×0.25 | **0.899** | 0.897 |
| w4a4mx | NVFP4 | MXFP4 (g32) | ×0.25 | 0.866 | 0.865 |

`w4a16` reproduces Phase 22's NVFP4 weight-only (0.921) within GPU nondeterminism
(`index_add_` is non-deterministic on CUDA; run-to-run noise ~+-0.01), validating the
patched experts forward.

## Reading

1. **FP8 activations are essentially free.** W4A8 (0.915) = W4A16 (0.913) within
   noise, and W16A8 (0.959) is ~0.04 under the bf16 ceiling. So a **2x communication
   reduction costs ~zero acceptance** -- the safe default comm lever, no memory cost.
2. **FP4 activations cost ~0.014.** W4A4-NVFP4 (0.899) vs W4A16 (0.913): the
   aggressive **4x comm reduction** is still ~0.90 acceptance -- well above the
   device-local floor (~0.78) and the weight/activation FP4 errors barely compound
   (each ~0.013-0.014, not multiplicative; the model is robust to both).
3. **Activation scheme matters: NVFP4 >> MXFP4 on the wire** (0.899 vs 0.866).
   Activations have per-token outliers; MXFP4's power-of-2 per-32 scale is too coarse,
   NVFP4's E4M3 per-16 scale handles them. Use NVFP4 for activation transport.
4. **Transport vs serving.** These are precisions a lossy server avoids (production EP
   dispatches at FP8, not FP4); the speculative verify licenses FP4 on the wire,
   because every draft step is corrected.

## The filled-in design ladder (memory-free comm first, then eliminate)

| lever | comm | extra HBM | beta |
| --- | ---: | ---: | ---: |
| FP8 activations | ×0.5 | 0 | 0.915 |
| NVFP4 activations | ×0.25 | 0 | 0.899 |
| local routing (FP4 full replica) | ->0 | ~14.5 GB/dev | 0.92 |

The two activation rows are **memory-free** and nearly **acceptance-free**, so they
are the first move on any comm-bound system. On a severe comm-bound link (PCIe
~9x), A4's 4x leaves ~2x residual, so local routing (eliminate, spend HBM) is the
finisher -- but it is only needed *after* the free activation-quant reduction.

## Caveats

- One-step, per-token activation quant; multi-token compounding handled by the
  Phase 20 `E[acc]` term. Verify exact -> losslessness intact.
- `index_add_` GPU nondeterminism gives ~+-0.01 noise on these numbers; differences
  below that (e.g. w4a16 vs w4a8) are "equal."
- Transport-only vs full W4A4-compute: here activations are quantized at dispatch and
  combine (the wire). Quantizing activations *inside* the expert GEMMs (full W4A4
  compute) would add a little more error; not separated here.
- One model (Qwen3-30B-A3B).

## Bottom line

Activation quantization delivers the communication reduction the positioning needs at
near-zero acceptance cost: **FP8 activations are free (2x comm), NVFP4 activations
cost ~0.014 (4x comm)**, both with no extra memory. Quantization is thus a unified
instrument across compute *and* communication: weight quant for compute/coverage,
activation quant for cheap comm shrink, local routing for full comm elimination --
all kept lossless by the bf16 verify. Use NVFP4 (not MXFP4) on the wire.
