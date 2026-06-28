# Results: Local-Routing Strategy -- shared-expert anchor + FP4 cache-size curve

Date: 2026-06-27. Real-weight rejection-sampling acceptance vs the bf16 full-routing
target (Phase 18/22 method; 16 prompts, decode positions, 512 samples). Local routing
= the comm-free draft (router restricted to a per-layer top-C local cache by request
gate mass). The router patch matches the real router exactly (softmax over all experts
-> restrict the *selection* to the local set -> top-k -> optional norm), so at C=E it
reduces to the target (beta=1), independent of `norm_topk_prob`.

## Exp 1: a shared-expert anchor lifts local routing (the untested lever)

Qwen1.5-MoE-A2.7B (60 routed, top_k 4, + a large sigmoid-gated shared expert).
**Shared expert carries ~45% of the MoE-output norm.** Same model/prompts, two modes:
`with_shared` (draft = shared + local-routed; verify = shared + full-routed) vs
`without_shared` (shared ablated in both).

| routed C/E | with shared | without shared | lift |
| ---: | ---: | ---: | ---: |
| 0.067 (4) | **0.376** | 0.150 | **+0.226 (2.5x)** |
| 0.133 (8) | 0.445 | 0.411 | +0.034 |
| 0.25 (15) | 0.611 | 0.520 | +0.091 |
| 0.5 (30) | **0.737** | 0.606 | +0.131 |
| 0.75 (45) | 0.843 | 0.732 | +0.111 |
| 1.0 (60) | 1.000 | 1.000 | 0 (sanity: full coverage -> exact) |

**The always-local shared expert dilutes the routed-coverage error**, lifting
acceptance by ~0.09-0.13 across cache sizes and up to **2.5x at tiny cache** (0.067E:
0.38 vs 0.15). To reach beta~0.74, *without* shared needs ~0.75E routed cache;
*with* shared, only ~0.5E. This is the one local-routing lever never tested
(Phase 08 scope was no-shared-expert) -- and it is a real, structural relaxation of
the coverage/replication requirement, because the shared expert is **EP-invariant
(always local)**, escaping the R~cN scaling.

## Exp 2: FP4 cache-size curve -- local routing interpolates to the quant floor

Qwen3-30B-A3B (128 experts, top_k 8), local routing swept over cache C, bf16 vs NVFP4
weights.

| C/E | bf16 | NVFP4 |
| ---: | ---: | ---: |
| 0.0625 (8) | 0.221 | 0.242 |
| 0.125 (16) | 0.398 | 0.400 |
| 0.25 (32) | 0.619 | 0.598 |
| 0.5 (64) | 0.846 | 0.840 |
| 0.75 (96) | 0.934 | 0.900 |
| 1.0 (128) | **1.000** | **0.920** |

Local routing interpolates from coverage-limited (small C) to the cap: **bf16 -> 1.0**
at full coverage (sanity: draft == verify), **NVFP4 -> 0.92** (the quant floor, =
Phase 22 weight-only). The curves track closely until C>=0.75E, where NVFP4 plateaus
at the quant floor while bf16 continues to 1.0. So in the quantized setting beta is
`~min(coverage(C), quant)`: to get high beta you scale the local cache toward full,
which FP4 makes affordable (Phase 18/21 pivot).

## Synthesis: the comm-free draft has two ways to raise beta

The comm-free local-routing draft (best for comm-bound serving, Phase 24) is
beta-limited; both experiments give it a lever:
1. **Shared-expert anchor** (Exp 1): for shared-expert models the always-local shared
   mass lifts beta by ~0.1 (more at small cache), at no replication cost. Lift scales
   with the shared mass fraction.
2. **FP4 cache scaling** (Exp 2): scale the local cache toward full -> beta -> 0.92,
   affordable under FP4.

Combined (a shared-expert model + FP4 local cache) is the strongest comm-free draft:
shared anchor lifts the curve, FP4 makes a large routed cache cheap.

## Follow-up: cross-architecture confirmation (DeepSeek-V2-Lite) + alpha-proxy (failed)

**DeepSeek-V2-Lite** (64 routed top_k 6 + 2 shared, norm_topk_prob=False, native
transformers impl) -- a *different architecture*. Shared mass fraction **~48%**
(similar to Qwen1.5-MoE's 45%, not smaller as expected: 2 shared experts are large).

| routed C/E | with shared | without shared | lift |
| ---: | ---: | ---: | ---: |
| 0.0625 (4) | 0.261 | 0.006 | **+0.255 (43x)** |
| 0.125 (8) | 0.388 | 0.014 | +0.374 |
| 0.25 (16) | 0.627 | 0.051 | **+0.576 (12x)** |
| 0.5 (32) | 0.822 | 0.237 | +0.585 |
| 0.75 (48) | 0.903 | 0.397 | +0.506 |
| 1.0 (64) | 1.000 | 1.000 | 0 (sanity) |

The shared anchor generalizes across architectures and is **even larger** here: it is
the difference between local routing being **useless** (0.006-0.05 at small cache,
much lower than Qwen's 0.15 -- DeepSeek's routed experts are more specialized) and
**usable** (0.26-0.63). Confirms the mechanism on a second architecture.

**alpha-proxy (invalid).** Scaling Qwen1.5-MoE's shared output by alpha in [0,1] to
emulate a smaller native shared fraction *failed*: the endpoints alpha=0/alpha=1
reproduce Exp 1 exactly, but intermediate alpha pushes the model out-of-distribution
(alpha=0.5 -> beta=0.27, a *negative* lift; phi non-monotonic). Lesson: you cannot
emulate a smaller shared fraction by scaling a 45%-mass component -- the residual
balance breaks. The small-fraction regime needs a natively-trained model.

**Third architecture (Moonlight-16B-A3B, DeepSeek-V3 sigmoid + noaux_tc gating,
native impl).** Shared mass fraction **~65%** (even higher: the
routed_scaling_factor=2.446 hypothesis was wrong -- sigmoid routing weights are small
[0,1], so the large shared MLP dominates the norm even more, not less).

| routed C/E | with shared | without shared | lift |
| ---: | ---: | ---: | ---: |
| 0.0625 (4) | 0.228 | 0.102 | +0.126 |
| 0.125 (8) | 0.368 | 0.135 | +0.233 |
| 0.25 (16) | 0.589 | 0.207 | **+0.382** |
| 0.5 (32) | 0.837 | 0.297 | **+0.540** |
| 0.75 (48) | 0.943 | 0.459 | +0.484 |
| 1.0 (64) | 1.000 | 1.000 | 0 (sanity) |

Confirms the anchor on a **third architecture** (V3 sigmoid gating), +0.13 to +0.54.
Cross-model, the lift tracks BOTH the shared fraction AND routed specialization: the
DeepSeek family (V2-Lite, Moonlight) has specialized routed experts whose
without-shared local routing collapses to ~0 at small cache -> huge lift; Qwen's
routed experts are less specialized (without-shared starts at 0.15) -> smaller lift.

**What remains genuinely untested (hardware-gated):** all three *runnable* shared-
expert MoEs turn out to be **high shared fraction (45-65%)**. The small-fraction
regime (DeepSeek-V3 1-shared ~11%, GLM-4.5) lives only in **frontier-scale** models
not runnable on one H100 -- so "small shared -> smaller lift" stays a prediction (the
mechanism makes it likely), confirmable only on a multi-GPU / rental testbed. The
anchor mechanism itself is now robust across 3 architectures and 2 gating types.

## Caveats

- **Shared lift is model-specific.** Qwen1.5-MoE's shared expert is unusually large
  (intermediate 5632 = 4x routed, ~45% mass). DeepSeek-V3 (1 shared : 8 routed active)
  has a smaller shared fraction (~10-15%) -> a proportionally smaller (but still real)
  lift. Generality needs a DeepSeek/GLM/Llama-4-class shared-expert model.
- One-step sampled acceptance proxy (multi-token via the Phase 20 `E[acc]` term);
  verify exact -> losslessness unaffected.
- The router patch is validated by the C=E -> 1.0 (bf16) / 0.92 (NVFP4) sanity points.
  (An earlier logits-masking patch silently mis-normalized weights for
  `norm_topk_prob=False` models -- caught by the C=E sanity check and fixed.)

## Bottom line

Local routing -- the comm-free draft -- is not stuck at the ~0.8 coverage cap. A
**shared-expert anchor** raises it by ~0.1 (up to 2.5x at small cache) for shared-expert
models, and **FP4 cache scaling** takes it to 0.92 at full coverage. The strongest
comm-free draft combines both.
