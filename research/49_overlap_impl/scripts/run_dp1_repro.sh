#!/usr/bin/env bash
# OV1(b2) DP1 deterministic repro: Qwen1.5-MoE-A2.7B on GPU 0 only, batch 2,
# short greedy generation, with W7_B2_DUMP writing the per-cycle aligned
# event stream (run: anchor/p1/outputs; consume: verdict b/rej/committed +
# served drafts). One pass of the resulting table should pin the recovery
# defect with the depth asymmetry now understood.
set -u
REPO=/data/smcho/self-spec-moe
cd "$REPO"
export PYTHONPATH="$REPO"
export PATH="$REPO/.venv/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0
export OV0B_MARK=1
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
export VLLM_SELF_SPEC_DRAFT_FULL_REPLICA=1 VLLM_SELF_SPEC_DRAFT_LOCAL_ROUTE=1
export VLLM_SELF_SPEC_DRAFT_FULL_CG=1 VLLM_SELF_SPEC_COMPILE_CONSISTENT=1
export VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1
export VLLM_SELF_SPEC_DRAFT_GRAPH_POOL=1
export VLLM_SELF_SPEC_DRAFT_WORKSPACE=1
export VLLM_SELF_SPEC_AHEAD_CHAIN=${AHEAD:-1}
export VLLM_SELF_SPEC_CONSUME_AHEAD=${AHEAD:-1}
export W7_MODEL="$HOME/.cache/huggingface/hub/models--Qwen--Qwen1.5-MoE-A2.7B/snapshots/1a758c50ecb6350748b9ce0a99d2352fd9fc11c9"
export W7_DP=1 W7_TP=1 W7_TRC=0 W7_EAGER=0
export W7_ITERS=1 W7_WARMUP=0
export W7_KS=2 W7_BATCHES=2
export W7_OUTLEN=48 W7_SHORTLEN=8
export W7_OUT="$REPO/research/49_overlap_impl/data"
export W7_TAG="${TAG:-dp1repro}" W7_DRAFT_QUANT=fp8 W7_A2A_US=0
export W7_B2_DUMP="$REPO/research/49_overlap_impl/data/${TAG:-dp1repro}_dump"
rm -f "$W7_B2_DUMP".rank*.jsonl

LOGD="$REPO/research/49_overlap_impl/logs"
mkdir -p "$LOGD"
timeout 3000 "$REPO/.venv/bin/python" \
  "$REPO/research/34_worldA_system/scripts/w7_fp8_timing.py" spec \
  > "$LOGD/${TAG:-dp1repro}.log" 2>&1
echo "EXIT=$?"
pkill -9 -f "w7_fp8_timing" 2>/dev/null
true
