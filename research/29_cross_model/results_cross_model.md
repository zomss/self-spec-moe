# Results: cross-model validation of the batched-memory enablers

Date: 2026-06-29. Routed-expert routing skew on three MoE families (`cross_model_skew.py`).

## Union growth (per-device resident requirement) -- UNIVERSAL

Distinct routed experts B concurrent requests need (per layer, % of routed E):

| B | Qwen3 (E=128) | DeepSeek-V2-Lite (64 routed) | GPT-OSS-20B (E=32) |
| ---: | ---: | ---: | ---: |
| 1 | 6% | 9% | 12% |
| 2 | 12% | 18% | 23% |
| 4 | 20% | 31% | 39% |
| 8 | 32% | 51% | 57% |
| 16 | 47% | 73% | 75% |
| 32 | 61% | 89% | 88% |
| 64 | 72% | 97% | 95% |
| 128 | 80% | 100% | 97% |

The Phase 28 batched-memory finding GENERALIZES: the per-device union grows with batch and
saturates near the full routed set by B~32-128 on every model. **It saturates FASTER
(relative to E) for fewer-expert models** (DeepSeek 64, GPT-OSS 32 hit ~88-89% by B=32 vs
Qwen3's 61%), because top_k/E is larger. So the per-request small-cache win is a low-batch
property everywhere, and high-batch needs ~the full routed set -- more so for small-E models.

## Global-hot coverage (skip-cold beta proxy) -- skew holds, magnitude varies

Fraction of a request's routed top_k that a fixed globally-hot top-C cache covers:

| cache C | Qwen3 | DeepSeek (routed) | GPT-OSS |
| ---: | ---: | ---: | ---: |
| 6% E | 0.26 | 0.15 | 0.16 |
| 12% E | 0.42 | 0.26 | 0.29 |
| 25% E | 0.66 | 0.44 | 0.50 |
| 50% E | **0.92** | **0.70** | **0.78** |

All three are skewed (a half-size cache covers most routing), so the globally-hot +
skip-cold design is viable across families -- but **Qwen3 is the most skewed** (0.92 @
50%); DeepSeek/GPT-OSS routing is flatter (0.70-0.78 @ 50%).

## The shared-expert anchor (DeepSeek) is the key differentiator

DeepSeek's ROUTED experts look worst for caching (flattest coverage 0.70 @ 50%, fastest
union saturation 89% @ B=32). BUT it has **2 always-resident SHARED experts (free,
batch-independent) carrying ~48% of the MoE-output mass** (Phase 25). So the effective
skip-cold beta = shared (free, ~half the mass) + routed-cache coverage -> a small routed
cache suffices because the shared experts already cover half the output. This matches
Phase 25 (shared anchor lifts DeepSeek local-routing beta by +0.25 to +0.58). So
shared-expert architectures have a MORE favorable memory story: a big mass chunk is free
and batch-independent.

## Bottom line

The Phases 26-28 enablers are NOT Qwen3 artifacts:
- batched-memory union growth -> universal (worse for small-E models).
- routing skew enabling a globally-hot cache -> holds across families (Qwen3 most skewed).
- shared-expert anchor -> a free, batch-independent mass floor that eases the memory story
  for DeepSeek-class (shared-expert) models.

Scope: skew enablers validated cross-model; the beta refinements (renorm recovery, dynamic
last-token cache, depth decay) remain Qwen3-measured, expected to generalize.
