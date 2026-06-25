# Phase 22: FP4 Expert-Draft Acceptance

Source: Phase 18 (FP8 draft acceptance = 0.954; method reused), Phase 21 (per-node
streaming/replication, which needs FP4 to fit), intra-node-EP draft direction.

## Objective

The per-node-replication / intra-node-EP draft only closes the **memory** math at
FP4 (the full quantized expert set must fit per node). Phase 18 proved FP8 is a
near-perfect draft (0.954); the open question is whether **FP4** -- the bit-width the
memory math actually requires -- still accepts well enough to draft.

```
Is an FP4-expert draft still a strong draft against the bf16 verify target?
```

## Method

`fp4_acceptance.py` fake-quantizes (quant->dequant) the expert weights weight-only
(W4A16) under several schemes, then measures rejection-sampling acceptance vs the
exact bf16 full-routing target at many teacher-forced positions (same metric as
Phase 18). Schemes: `fp8_pertensor` (anchor), `mxfp4_g32` (E2M1 + power-of-2/32
scale), `nvfp4_g16` (E2M1 + E4M3/16 scale), `int4_g128` (symmetric baseline).

```bash
V=/data/smcho/self-spec-moe/.venv/bin/python
CUDA_VISIBLE_DEVICES=0 $V fp4_acceptance.py --model Qwen/Qwen3-30B-A3B \
  --local-files-only --output-json data/qwen3_30b_fp4_acceptance.json
```

## Decision criterion

FP4 is viable for the direction if acceptance stays well above the device-local
floor (~0.78-0.85) and close enough to FP8 (0.95) to keep the Phase 20 speedup
strongly net-positive.

## Result (see `results_fp4_acceptance.md`)

FP4 acceptance is **0.90-0.92** (NVFP4 best at 0.921, MXFP4 0.905), only ~0.05 below
the FP8 anchor (0.967 here) and far above local routing. Translates to ~2.0-2.1x
lossless at the Phase 20 high-exposure point (vs FP8's 2.3x). **FP4 is small enough
to replicate per node *and* accurate enough to draft** -- the acceptance side of the
intra-node-EP direction holds. NVFP4 (Blackwell-native) is the target format; MXFP4
is the ~1.5-pt-lower fallback.

## Next artifact

Pairs with the per-node memory-fit table (models x node-counts x bit-widths): this
phase fixes the acceptance at each bit-width; that table fixes what fits. Together
they bound which targets the intra-node-EP draft is viable for.
