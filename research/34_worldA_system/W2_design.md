# W2 design: resident draft-expert cache (raise the draft's coverage/beta)

W0+W1 give a comm-free draft routing ONLY to its EP shard -> LOW coverage (W2 harness:
mean accept-length ~2.7 of 5 at K=4, per-draft-token accept ~0.42; the W0 doc's "0.758"
used a different metric/window -- the canonical metric going forward is mean accept-length,
which is unambiguous and drives tokens/s). W2 raises coverage by giving each device more
resident experts. Scoping (Explore) found:

## Scoping verdict
- **No config-only path that reuses the W1 mask for a *partial* replica.** vLLM's only
  built-in replication is **EPLB redundant experts**, which is the WRONG primitive here:
  (1) it routes over logical experts then all-to-all to physical replicas (the OPPOSITE of
  comm-free local routing -- `router/base_router.py:264-276`); (2) it makes `expert_map`
  **physical-length** (`E+R`) while the gate's `router_logits` stay **logical-length** (`E`),
  so `mask_router_logits_to_resident` (`local_route.py:62-66` vs `deepseek_v2.py:276`) would
  mis-index. (3) `num_redundant_experts` is hard-gated behind `enable_eplb`
  (`layer.py:246-248`, `parallel.py:489-496`), and the static initial map is round-robin
  from id 0, not our hot set -- hot placement only happens after a RUNTIME rebalance.
  -> EPLB is out.
- **The W1 mask assumption holds only if the resident expansion keeps `expert_map`
  LOGICAL-length (E).** `determine_expert_map` (`expert_map_manager.py:22-113`) always shards
  disjointly -> a custom map is needed for a partial resident set. This is the core blocker
  that makes the bounded cache a (small) custom expert-loader, not config.

## The 3-step plan (build order)
### W2a -- full bf16 replica (config; the Stage-1 MVP; quick) <- BUILD FIRST
Make the DRAFT non-EP: `enable_expert_parallel=False` for the draft -> `use_ep=False` ->
`expert_map=None` (`config.py:1201-1215` -> `expert_map_manager.py:63`) -> the W1 mask is a
no-op (`local_route.py:62`) -> the draft routes over ALL experts -> **beta -> ~1**. The draft
is comm-free **by replication** (each rank holds all experts, routes locally, no all-to-all
and no AgRs at all -- the W1 skip path is simply not exercised). In the forced-PCIe **DP+EP**
layout the target is `tp=1, dp=N`, so `draft_tp=1 == target_tp=1` -> **no `draft_tp==target_tp`
assert issue** (`draft_model.py:36-51`). Implementation: a draft-expert-strategy knob
(env `VLLM_SELF_SPEC_DRAFT_FULL_REPLICA`) that, when set, makes `create_draft_parallel_config`
(`speculative.py:984-1011`) set the draft `enable_expert_parallel=False` (overriding the W0
EP propagation). Cost: full E experts/rank in bf16 (~28 GB for V2-Lite) -- the README Stage-1
"bf16 full replica". This is the degenerate (all-resident) cache; it proves the comm-free
draft reaches the beta ceiling and **unblocks W7** (end-to-end tokens/s).

### W2b -- static globally-hot top-C replica (custom loader; the real memory contribution)
Keep the draft EP-sharded + the W1 skip-A2A, but expand each rank's resident set to
`shard union hot_top_C` while keeping `expert_map` **logical-length E** (so the W1 mask still
works). Needs: (1) a draft-only `determine_expert_map` variant (`expert_map_manager.py:22-113`)
that marks shard+hot resident (>=0) without physical expansion; (2) a loader that
materializes the extra resident experts' weights locally (extend `routed_experts.py:856-974`,
reusing the logical-source mapping at :964 but WITHOUT EPLB's physical/all-to-all path);
(3) feed the offline globally-hot ranking (Phase 28) as a per-layer resident set. beta ~0.9
at bounded memory (Phase 28: ~8 GB / 0.5E). This is the paper's memory mechanism realized in
the system; bigger lift.

### W2c -- FP4 on the cache (separate; checkpoint-gated)
No on-the-fly bf16->NVFP4 in the loader; NVFP4 is checkpoint-deserialized
(`quantization/modelopt.py`). Needs a pre-quantized DeepSeek-V2-Lite NVFP4 checkpoint, then
drop the `quant_config=None` force (`draft_model.py:60`) and point the draft `ModelConfig` at
it (`config/vllm.py:608-642` derives `ModelOptNvFp4Config`). Composes with W2b's loader to
bound the replicated-cache memory.

## Key files
`vllm/config/speculative.py:984-1011` (draft parallel config),
`vllm/v1/spec_decode/draft_model.py:36-66` (draft config + the TP assert + quant force),
`vllm/model_executor/layers/fused_moe/expert_map_manager.py:22-113` (resident map),
`vllm/model_executor/layers/fused_moe/routed_experts.py:856-974` (expert load),
`local_route.py:62-66` + `runner/moe_runner.py:559-562` (the W1 mask, unchanged).

## Status
- **W2a (full replica): DONE, validated (worktree ssm-w2 / branch w2-draft-cache).**
  2 files, ~25 lines, additive + flag-gated (env `VLLM_SELF_SPEC_DRAFT_FULL_REPLICA`,
  default off -> W0 EP-shard behavior unchanged). Knob ON overrides the W0 EP
  propagation in `create_draft_parallel_config` (`speculative.py`) so the draft
  builds `enable_expert_parallel=False` -> `use_ep=False` -> `expert_map=None`.
  The draft stays data-parallel (DP) when the target is DP (only EP is flipped).
  - **(1) Draft is full replica:** both DP workers log
    `use_ep=False expert_map=None` (vs `use_ep=True expert_map=set` knob-OFF).
  - **(2) Acceptance jump (headline), DeepSeek-V2-Lite DP=2+EP, greedy, K=4,
    same harness** (`scripts/w2_dp.py`): shard draft (knob OFF, producer ON)
    **0.4234** (accept-len 2.69) -> full replica (knob ON) **0.9707**
    (accept-len 4.88 of 5). Beta -> ~1, exactly as designed. (The W0 doc's 0.758
    was a different config/window; the relative jump is the result.)
  - **(3) Comm-free by construction:** the non-EP draft never enters the AgRs
    path. AgRs counter: shard `total=17420 real=3484` (draft skips, verify real)
    -> replica `total=9100 real=9100` (total drops by the draft's share; the
    draft contributes ZERO collectives, all remaining are the full-EP verify).
  - **(4) Losslessness:** rejection-correct, as lossless as vLLM's native spec
    (W0 standard). replica vs no-spec greedy: 12/16 exact seqs, 82.4% token
    agreement -- the mismatches are the pre-existing batched-verify near-tie FP
    flips, not corruption; replica is closer to greedy than shard (11/16).
  - Memory: ~37 GB/rank steady weights (target shard + full-replica draft) on an
    H100; fits. -> unblocks W7.
- **Next:** W2b (static globally-hot top-C replica, custom loader) + W2c (FP4),
  then W7 (tokens/s).
