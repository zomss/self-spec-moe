# Phase 54 results — node-local draft: acceptance validated, execution mode open

**Implementation** (branch working tree, env-gated, default off):
`VLLM_SELF_SPEC_DRAFT_NODE_LOCAL=1` -- draft routes to any expert resident on
its NODE (union residency mask, `determine_node_expert_mask`) and
dispatches/combines over a new intra-node `dp_node` subgroup (NVLink-only
AgRs, node-sliced sizes); verify unchanged (global EP). Files:
`envs.py, forward_context.py, parallel_state.py, expert_map_manager.py,
moe_runner.py, all2all.py, llm_base_proposer.py`.

## 1. The test ladder: acceptance side FULLY VALIDATED

| step | config | routed coverage | accept @K=1 | prediction |
|---|---|---:|---:|---:|
| 1 | V2-Lite, 1 node DP8 | 100% (node=all) | **1.982** | ~1.9+ (invariant) |
| 2 | V2-Lite, 2 nodes DP16 | 50% | **1.878** | 1.88 (measured EP2 point) |
| 3 | DeepSeek-V2 236B, 2 nodes | 50% | **1.905 / 1.847** (2 runs, b8) | 1.85-1.9 |

The coverage->acceptance curve is now measured in-engine end to end:
6.25% -> 1.36 (Phase 53 preview), 12.5% -> 1.36, 50% -> ~1.88, 100% -> 1.98,
and it TRANSFERS to 236B unchanged. The node-local draft breaks the
acceptance-vs-EP anti-correlation exactly as the Phase 11/19 design intended:
at N nodes the draft sees 1/N of experts by NODE, not 1/EP by device.

## 2. Open issue 1: draft cycle inflation -- DIAGNOSED (not eager; rendezvous drain)

Node-local wall-clock is far below the cycle model: 236B b8 = 68 tok/s vs
264 baseline (cycle ~217 ms vs ~55-60 predicted); V2-Lite cycles 32 (1-node)
/ 56 ms (2-node) vs device-local 23 ms.

**Torch-trace A/B (V2-Lite DP8 1-node, node-local vs device-local):**
launch profiles are comparable (332 vs 410 graph launches, 1.5k vs 1.9k
kernel launches -- NOT an eager storm) and the blocking-sync COUNT is
identical (460 vs 466 aten::item), but the sync WAITS inflate 786 vs 535 ms
(max 19 vs 12 ms). The draft's real intra-node collectives lengthen the
drain at every existing DP-coordination rendezvous; the per-chain-step
coordination (dp_utils min().item(), Phase 52's finding) is the multiplier.

**DRAFT_SKIP_DP_COORD x node-local: UNSTABLE** -- CUDA illegal instruction
during graph capture (subgroup PG): with coordination skipped, ranks can
capture divergent shape sequences around the node collectives (the exact
deadlock class the OV1 consume-mode comment warns about). Flag combination
unsupported until fixed.

**Fix direction (next session):** amortize instead of skip -- ONE
coordination per spec cycle (covering all chain steps + verify) instead of
one per forward; or node-scoped coordination (the node group is the only
consistency domain the node collective needs). Both keep capture coordinated.

## 3. Open issue 2: mixed-step dp_metadata crash at b>=32 (see Phase 53 §3)

`AssertionError: 1 != 203`: the draft proposes during mixed prefill/decode
steps with stale DP-padded metadata; any GATHERING draft crashes (device-
local is immune only because it skips the gather). A locally-uniform sizes
fallback was added for the node branch (uniform-batch scope), but the crash
site is the DEFAULT path inside `naive_dp_ep.prepare` (the FP8 modular
draft), reached by a drafter sub-forward that lacks the spec context key.
Proper fix: drafter forwards must carry their own coordinated sizes.

## 4. Where this leaves the thesis

With f ~0.5 measured at 236B (Phase 53) and accept 1.9 measured for the
node-local draft at the same scale, the cycle model gives
`accept / ((1-f)*q_draft + verify_ratio)` ~= 1.9 / (0.5*0.8 + 1.5) ~= 1.0-1.15
at K=1 -- break-even to modest win IF the draft executes at its modeled cost.
Both open issues are execution-path engineering, not physics or acceptance.
The measured pieces (f at scale, coverage curve, node-local mechanism) are
the paper's load-bearing numbers either way.
