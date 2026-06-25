# Results: FP4 Expert-Draft Acceptance vs bf16 Target

Date: 2026-06-25. Real Qwen3-30B-A3B (E=128, top_k=8, 48 layers). Expert weights
fake-quantized (quant->dequant) **weight-only** (W4A16; attention/router/embeddings
stay bf16). Rejection-sampling acceptance against the exact bf16 full-routing target
`p`, at 122 teacher-forced positions (16 prompts), 512 samples/position. Same method
as Phase 18 (whose per-tensor FP8 gave 0.954 -- reproduced here as the sanity anchor).

## Acceptance

| scheme | bits/wt | sampled acc | expected | top-1 |
| --- | ---: | ---: | ---: | ---: |
| fp8_pertensor (E4M3) | 8 | **0.9671** | 0.9673 | 0.9672 |
| nvfp4_g16 (E2M1 + E4M3/16 scale) | ~4.5 | **0.9209** | 0.9216 | 0.9180 |
| mxfp4_g32 (E2M1 + E8M0/32 scale) | ~4.25 | 0.9054 | 0.9051 | 0.8852 |
| int4_g128 (symmetric) | ~4.06 | 0.9014 | 0.9011 | 0.8770 |

- FP8 anchor 0.967 ~ Phase 18's 0.954 (122 vs 16 positions) -> method consistent.
- **FP4 acceptance is 0.90-0.92**, only ~0.05 below FP8 and far above the
  device-local routing floor (~0.78-0.85, Phases 13/15). **NVFP4 is the best FP4**
  (0.921) -- its finer E4M3 per-16 block scale beats MXFP4's power-of-2 per-32 scale
  by ~1.5 pts, matching the weight-error gap (NVFP4 9.6% vs MXFP4 12.5%).

## What this means for speedup (Phase 20 model)

`speedup(k) = E[acc](k,beta) / ((k+1) - k*f)`, `E[acc]=(1-beta^(k+1))/(1-beta)`, at
the Phase 20 low-batch operating point `f=0.723` (163 us/coll):

| draft beta | best-k speedup @ f=0.723 | accepted/cycle (k=8) |
| --- | ---: | ---: |
| 0.95 (FP8) | 2.30x | ~6.6 |
| **0.921 (NVFP4)** | **2.08x** | ~5.6 |
| 0.905 (MXFP4) | 1.95x | ~5.1 |

FP4 keeps the draft in the high-acceptance regime: ~5-6 accepted tokens per k=8
cycle and ~2x lossless speedup at the high-exposure point -- ~10% below FP8, still
strongly net-positive, and well clear of break-even.

## Why this matters for the direction

The per-node-replication / intra-node-EP draft (Phase 21 + intra-node EP design)
only closes the **memory** math at FP4 (full quantized expert set must fit per node).
This phase confirms FP4 also closes the **acceptance** side: an FP4 expert draft is
still a strong draft. So the two constraints are jointly satisfiable -- FP4 is small
enough to replicate per node *and* accurate enough to draft well. NVFP4 (Blackwell-
native) is the format to target; MXFP4 (~0.905) is the Hopper/GPT-OSS fallback,
~1.5 pts lower but still fine.

## Caveats

- **One-step, weight-only proxy.** Per-position sampled acceptance; multi-token
  compounding over a cycle is captured by the `E[acc]` geometric term, not measured
  end-to-end here. Verify is exact bf16 -> losslessness is unaffected by draft quant.
- **W4A16 only.** Quantizing activations too (W4A4) for extra speed would lower
  acceptance; the design keeps bf16 activations, so W4A16 is the relevant number.
- **One model.** Qwen3-30B-A3B. GPT-OSS is natively MXFP4 (its FP4 draft is ~lossless
  vs its own deployment -- a degenerate comparison), so it is not a clean FP4-
  degradation test; a non-MXFP4-native second model would strengthen generality.
- NVFP4 costs ~0.25 bit/weight more than MXFP4 (E4M3 vs E8M0 scale) for ~1.5 pts
  acceptance -- a minor size/accuracy knob, both still "FP4" for the per-node fit.

## Bottom line

FP4 expert drafts accept at **0.90-0.92** (NVFP4 best, 0.921), ~0.05 below FP8 and
far above local routing. FP4 is simultaneously small enough for per-node replication
and accurate enough to draft -- the acceptance side of the intra-node-EP direction
is valid.
