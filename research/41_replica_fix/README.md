# Phase 41: fix Bug B — make the comm-free full-replica draft a GENUINE replica

**Source phase:** 40 (num-divergence). Phase 40 pinned the comm-free full-replica
draft's large-EP accept collapse to TWO bugs. Bug A (override discarded) was fixed
on this branch (draft now 60/60 experts, `use_ep=False`). Bug B remained: with
`use_ep=False` in a pure-DP deployment, `FusedMoEParallelConfig.make` takes the
non-EP branch -> `flatten_tp_across_dp_and_pcp` sets `tp_size = dp_size*tp_size`
(=8 at DP=8) -> TP-shards each expert (2816->352) and relies on
`tensor_model_parallel_all_reduce` to recombine. But in pure-DP
`get_tp_group().world_size == 1` -> that all-reduce is a NO-OP -> each rank emits a
1/dp_size-sharded MoE output. Accept stuck at 2.18 (DP=8).

## The fix
`vllm/model_executor/layers/fused_moe/config.py` `FusedMoEParallelConfig.make`:
add a full-replica branch (gated on `VLLM_SELF_SPEC_DRAFT_FULL_REPLICA`) that,
when non-EP with `dp_size*pcp_size > 1`, returns `tp_size=tp_size_ (=1),
ep_size=1` WITHOUT the flatten-TP. Each rank then holds FULL unsharded experts
and computes its own tokens locally — no TP-shard, no all-reduce. Default off ->
unchanged.

## Setup
`Qwen/Qwen1.5-MoE-A2.7B` (60 experts top-4, 24 layers, non-MLA), DP->EP,
forced-PCIe (`NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=1`), greedy,
K=4, bf16, `VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0`,
`DRAFT_FULL_CG=1 COMPILE_CONSISTENT=1`. Worktree `/data/smcho/ssm-num`, branch
`w7-num-divergence`. `PYTHONPATH=/data/smcho/ssm-num`.

## Gates
1. Replica is genuine: draft MoE-output rel-err ~0.0 vs plain MoE at DP=8;
   per-rank log shows full unsharded experts (tp=ep=1, w13 intermediate 2816).
2. Accept recovers: accept_len DP=2/4/8, K=4, greedy -> flat ~5.0 (from 4.88->2.18
   collapse).
3. Still comm-free (draft use_ep=False -> no all-to-all) + lossless (== the
   EP-full-draft reference, config C).

## Headline
`Qwen/Qwen3-30B-A3B`, DP=8/EP=8, FP8 full-replica draft, forced-PCIe,
FULL_CG+COMPILE_CONSISTENT: accept_len @ DP=8 AND decode tok/s speedup vs no-spec
(two-length-slope, `research/34_worldA_system/scripts/w7_fp8_timing.py`).

## Scripts
- `scripts/dump_moe.py` + `scripts/analyze_gate1.py` — Gate 1 (rel-err vs plain MoE).
- `scripts/probe_draft_weight_shape.py`, `research/40_*/scripts/probe_draft_tp_group.py`
  — Gate 1 structural (tp=ep=1, full experts).
- accept via `research/40_num_divergence/scripts/accept_run.py` (CFG=A/C) — Gate 2.
- `scripts/verify_lossless_cfg.py` — Gate 3 lossless (A==C).
- `scripts/run_headline.sh` (+ `w7_fp8_timing.py`) — headline.

## Output
`results.md` — fix, Gate 1/2/3, headline.
