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

**Amortization TESTED (Phase 55, `VLLM_SELF_SPEC_DRAFT_AMORTIZE_DP_COORD`):**
memoize coordinate_batch_across_dp per distinct shape per propose() (dummy
runs/capture always coordinate -- stable, unlike SKIP_DP_COORD). Verdict:
accept parity everywhere (1.973/1.885/1.916), **+10.7% single-node
(490->543 tok/s), ~0 cross-node** (270 vs 268 V2-Lite; 65 vs 68 @ 236B).
Together with the SKIP result this FALSIFIES the rendezvous-count hypothesis:
the wait relocates to the remaining sync points. **ROOT CAUSE CONFIRMED (profiled 236B b8 node-local):**
draft_forward_first (step0, the only draft forward at K=1) = 72.3 ms
(~all of draft_chain 75.9) vs ~10-15 ms compute; verify = 73.3 ms vs ~30 ms
no-spec step. Both are the SAME defect: q>1 / non-plain-decode shapes
dispatch PIECEWISE, so per-layer collectives + prepare/finalize glue run in
PYTHON between captured body pieces (~1 ms/MoE-layer x 59). The comm-free
device-local draft dodges it (its collectives are no-ops); any dispatching
draft AND the spec verify pay it. This unifies with the Phase 52
fabric-independent verify overhead -- one fix serves both:
**make spec forwards eligible for FULL cudagraph capture** so collectives
replay inside the graph like the plain decode step.

**Phase 55 attempt (STEP0_FULL_CG, landed but NOT ENGAGING):** the full
machinery is in (draft dispatcher keyed at q=K+1, q-aware capture-metadata
builder, step0 uniform dispatch + descriptor forwarding, runner capture-size
bypass; env-gated, accept parity everywhere, zero regressions). Per-shape
dispatch logging then showed WHY it cannot engage yet: steady-state step0 is
NOT per-request uniform -- shapes are (24 tok, 8 req)=3/req, (21,7), (15,5),
... The drafter pads the TOTAL to a bucket, not each request to a fixed q,
so no uniform-q FULL key ever matches. The real fix is one of:
(a) pad step0 per-request to a fixed q (EAGLE padded-drafter-batch style;
the landed capture machinery then engages as-is), or
(b) move the MoE dispatch/combine INSIDE the compiled piecewise body so
PIECEWISE replays the collectives on-graph (helps step0 AND all mixed
shapes, model-agnostic). (b) likely also fixes the Phase 52 verify anomaly
for chunked-prefill shapes. Both are drafter/compile surgery -- the next
session's task, now with exact shape evidence.

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
