# Phase 52: two_node_e2e — first REAL 2-node run of the comm-free self-spec

**Source phases:** 47 (fully-optimized stack: genuine FP8 full-replica comm-free
draft + PIECEWISE chain, K=2; single-node forced-PCIe win 1.03-1.30x, projection
1.6-2.1x at f~0.6-0.8), 48 (h107<->h106 fabric: 8x400G RoCEv2 rails + GDRDMA,
NCCL a2a ~70-115 us/coll at decode payloads -> predicted REAL 2-node win
**~1.2-1.4x at b32-64**, f~0.3-0.4).

**Objective.** Replace every emulated/forced-fabric input with a genuine
2-node measurement: Qwen3-30B-A3B, attention-DP16 + EP16 across h107+h106 over
real NVLink(intra)+RoCE(inter), spec (FP8 full-replica comm-free draft, K=2,
Phase-47 stack) vs no-spec, same two-length-slope decode methodology.
Also: single-node DP8 native-NVLink no-spec reference to bound the inter-node
comm fraction f directly (T_2node vs T_1node at matched per-rank batch).

**Assumptions.** Shared NFS repo+venv on both nodes; branch
`research/self-spec-moe` (HEAD 94cb8e33c, base = main 9037498c2, matches the
precompiled editable install). Model downloaded once to the shared HF cache.
NO forced-PCIe env (real fabric). NCCL: `NCCL_IB_HCA=^mlx5_8`,
`NCCL_IB_GID_INDEX=3`, bootstrap/gloo on `ens14np0`.

**Deltas vs the Phase-47 harness** (`w7_2node.py` is a phase-local copy of
`34_worldA_system/scripts/w7_fp8_timing.py`, which stays UNCHANGED):
- node-rank-aware spawning: global DP rank = node_rank*8 + local_rank,
  `VLLM_DP_RANK_LOCAL` = local rank (was = global rank);
- fixed pre-agreed master port per K (both nodes must compute the same port;
  the original used `get_open_port()`);
- only global rank 0 (node 0) collects/writes results, as before.

**Commands.** `scripts/run_2node.sh {nospec|spec}` on h107 (launches the h106
side over SSH; both source `scripts/env_2node.sh`);
`scripts/run_1node_ref.sh` for the native-NVLink DP8 reference.

**Decision criteria.** Compare measured 2-node spec/no-spec tok/s ratio at
b32/64/128 to the Phase-48 prediction (~1.2-1.4x at b32-64). accept_len should
stay ~2.9 (Phase 41/47: accept is fabric-independent). Report measured f
(1 - T_1node/T_2node per-rank-batch-matched) vs the predicted 0.3-0.4.

**Expected next artifact.** `results_two_node.md`: the first genuine
inter-node win table; then a DeepSeek-scale shared-expert model to move up the
f curve (Phase 53).
