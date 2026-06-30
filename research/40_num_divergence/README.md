# Phase 40: pin the numerical cause of the MoE accept divergence + the FP32 fix

**Source phase:** 38 (EP-isolation) + 39 (temperature). Phase 38 proved World A's
comm-free FULL-REPLICA draft accept COLLAPSES at large EP (DP=8 accept **2.18** vs
the EP all-to-all verify **5.00**), present even EAGER, after the compile bug is
fixed (`VLLM_SELF_SPEC_COMPILE_CONSISTENT=1`). The residual divergence is the
**MoE reduce STRUCTURE**: the comm-free draft sums ALL of a token's top-k experts
in ONE local bf16 `moe_sum`, while the EP verify sums each rank's SHARD in bf16
then bf16-reduces the partials cross-rank (`reduce_scatterv`). Same math, different
bf16 summation associativity -> divergence that worsens with EP width.

## Objective
1. **Step 1 — confirm + quantify.** Dump, for a fixed short prompt on DP rank 0 at
   one MoE layer, per token: (a) router logits, (b) selected expert ids+weights,
   (c) post-combine MoE output, (d) final next-token logits. Run config A
   (comm-free full-replica) and config C (EP all-to-all) on the SAME prompt; diff:
   confirm (a)+(b) MATCH (same routing), (c) DIVERGES (quantify rel error), (d)
   argmax flips (= the rejections).
2. **Step 2 — the FP32-accum fix.** Force FP32 accumulation in BOTH reduce paths
   (local `moe_sum` + cross-rank `reduce_scatterv`) so the structure stops
   mattering. Re-measure accept_len: does config A converge to ~5.0 at DP=8?
3. **Step 3 — cost.** FP32-accum decode tok/s overhead: cheap or expensive?

## Setup
`Qwen/Qwen1.5-MoE-A2.7B` (qwen2_moe, 60 experts top-4, 24 layers, non-MLA),
**DP=8 -> EP=8**, forced-PCIe (`NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0
NCCL_IB_DISABLE=1`), greedy, **K=4**, **bf16**. `VLLM_USE_DEEP_GEMM=0
VLLM_MOE_USE_DEEP_GEMM=0`. Accept runs: draft_model self-spec, `FULL_CG=1` +
`COMPILE_CONSISTENT=1`. Dump runs: EAGER main-model reduce-structure probe
(`enable_expert_parallel` + `VLLM_SELF_SPEC_LOCAL_ROUTE` toggle A vs C).

## Source changes (worktree /data/smcho/ssm-num, branch w7-num-divergence)
- Instrumentation (Step 1): `vllm/.../fused_moe/num_divergence_dump.py` +
  capture hooks in `runner/moe_runner.py` and `logits_processor.py`. Env
  `VLLM_SELF_SPEC_MOE_NUM_DUMP` (dir), `VLLM_SELF_SPEC_MOE_DUMP_LAYER`.
- FP32-accum fix (Step 2): env `VLLM_SELF_SPEC_MOE_FP32_ACCUM` gates fp32
  accumulation in `fused_moe.py` `moe_sum`, `topk_weight_and_reduce.py`, and
  `cuda_communicator.py` `reduce_scatterv`. Default off -> unchanged.

## Commands
```
# Step 1 dumps (one per config, DP=8):
CFG=A E_DP=8 .venv/bin/python research/40_num_divergence/scripts/dump_moe.py
CFG=C E_DP=8 .venv/bin/python research/40_num_divergence/scripts/dump_moe.py
.venv/bin/python research/40_num_divergence/scripts/analyze_dump.py        # bf16
.venv/bin/python research/40_num_divergence/scripts/analyze_dump.py --fp32  # fixed

# Step 2/3 accept + timing (A/C x bf16/fp32):
bash research/40_num_divergence/scripts/run_accept.sh
```

## Output
`results_W7_num_divergence.md` — Step 1 divergence table, Step 2 accept
before/after, Step 3 cost, verdict.
