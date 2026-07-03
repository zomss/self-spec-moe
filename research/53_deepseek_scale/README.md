# Phase 53: deepseek_scale — the f>=0.5 go/no-go on 2 real nodes

**Source phases:** 52 (Qwen3-30B on h107+h106: f=0.25-0.35, ideal-cycle ceiling
~0.9-1.0x -> the win needs f>=~0.45-0.5), 19 (the original multi-node plan:
DeepSeek-scale MoE, shared expert, local-routing draft + full-EP verify),
25/29 (shared-expert anchor makes local-routing drafts usable on the DeepSeek
family), 51 (fabric: 8x400G RoCEv2 rails, ~70-115us/coll at decode payloads).

**Objective.** Move f into the win regime by model scale, not fabric
degradation: DeepSeek-V2 (236B-A21B, 60 layers, hidden 5120, 160 routed + 2
shared experts, MLA) across h107+h106, attention-DP + EP16. Measure (1) the
no-spec step and its comm fraction at scale, (2) spec vs no-spec with the
at-scale draft.

**Design constraint (memory).** FP8 full-replica draft does NOT fit at 236B
(~118 GB/rank); the at-scale comm-free draft is the EP-SHARD local-routing
draft (DRAFT_LOCAL_ROUTE=1, no FULL_REPLICA) + always-local shared experts
(the Phase 25 anchor) + FP8 draft quant. Expected accept is the open risk:
anchor-lifted beta ~0.35-0.5 (Phase 25 V2-Lite) -> accept(K=1) ~1.35-1.5
against a win bar of ~(1-f)+verify_ratio ~1.6-1.75 at f=0.6-0.7. K=1
(MLA FULL-CG is K<=2-safe per Phase 35; Phase 50 found K*=1 for World B).

**Layout.** TP2 x DP8 = EP16 (TP2 halves the per-rank dense/draft memory;
target experts 444GB/16 + dense/2 + FP8 draft shard ~ 63-70 GB/rank).
Fallback TP4 x DP4. Same env stack as Phase 52 (performance governor, rail
gloo, no forced-PCIe knobs).

**Steps.**
1. Download deepseek-ai/DeepSeek-V2 (bf16, ~472 GB) to the shared NFS cache
   (background); DeepSeek-V2-Lite (~31 GB) for the smoke test.
2. Smoke (single-node, V2-Lite): draft_model + DRAFT_LOCAL_ROUTE on the
   deepseek_v2 arch + MLA draft chain; accept sanity vs Phase 25 ballpark.
3. 2-node V2-236B: nospec baseline b8/32/64; spec K=1 (and K=2 if stable).
4. Decision table: measured accept vs the f-dependent win bar.

**Decision criteria.** GO if spec/nospec > 1.0 at any batch; informative
either way: this is the first real measurement of BOTH sides (f at scale,
at-scale draft accept) of the Phase 19 gate on real hardware.

**Expected artifact.** `results_deepseek.md`.
