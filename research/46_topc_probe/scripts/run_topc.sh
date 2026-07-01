#!/usr/bin/env bash
# Phase 46 W7 top-C probe: sweep the DRAFT top-C prune (C = top_k .. 1) for a
# model and record accept_len(C) + draft_forward ms(C). Comm-free full-replica
# self-spec draft, piecewise chain on, forced-PCIe, K=2, greedy. One DP engine
# group per C, serial, tearing down OUR OWN workers between runs.
#
# Usage:  run_topc.sh <model_key> "<C list>"   e.g.  run_topc.sh qwen30b "8 7 6 5 4 3 2 1"
# Also runs a C=top_k nospec reference (TC_NOSPEC=1, one C) when NOSPEC=1 env set.
set -u
REPO=/data/smcho/ssm-topc
cd "$REPO"

export PYTHONPATH="$REPO"
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
export NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=1
# Full piecewise trio for the comm-free self-spec draft chain.
export VLLM_SELF_SPEC_DRAFT_FULL_REPLICA=1
export VLLM_SELF_SPEC_DRAFT_LOCAL_ROUTE=1
export VLLM_SELF_SPEC_DRAFT_FULL_CG=1
export VLLM_SELF_SPEC_COMPILE_CONSISTENT=1
export VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1
export VLLM_SELF_SPEC_LOG_A2A_COUNTS=1

MODEL_KEY="${1:?model_key required (qwen30b|qwen15moe)}"
CLIST="${2:?C list required}"

case "$MODEL_KEY" in
  qwen30b)
    export TC_MODEL=/home/smcho/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39
    export TC_TAG=qwen30b TC_TRC=0
    ;;
  qwen15moe)
    export TC_MODEL=/home/smcho/.cache/huggingface/hub/models--Qwen--Qwen1.5-MoE-A2.7B/snapshots/1a758c50ecb6350748b9ce0a99d2352fd9fc11c9
    export TC_TAG=qwen15moe TC_TRC=0
    ;;
  *) echo "unknown model_key $MODEL_KEY"; exit 1;;
esac

export TC_DP="${TC_DP:-8}" TC_TP=1
export TC_OUT="$REPO/research/46_topc_probe/data"
export TC_BATCH="${TC_BATCH:-64}" TC_K="${TC_K:-2}"
export TC_DRAFT_QUANT="${TC_DRAFT_QUANT:-fp8}"
export TC_GPU_MEM="${TC_GPU_MEM:-0.90}"
export TC_STEADY="${TC_STEADY:-80}" TC_WARMUP="${TC_WARMUP:-60}"

PY="/data/smcho/self-spec-moe/.venv/bin/python"
SCRIPT="$REPO/research/46_topc_probe/scripts/topc_sweep.py"
LOGD="$REPO/research/46_topc_probe/logs"
mkdir -p "$LOGD" "$TC_OUT"

kill_stragglers() {
  pkill -9 -f "topc_sweep" 2>/dev/null
  for pid in $(ps -u "$USER" -o pid,cmd | grep -E "EngineCore|VLLM::" | grep -v grep | awk '{print $1}'); do
    kill -9 "$pid" 2>/dev/null
  done
  sleep 6
}

run_c() {  # C  [nospec]
  local c="$1"; local nospec="${2:-0}"
  export TC_C="$c" TC_NOSPEC="$nospec"
  local tag="C${c}"; [ "$nospec" = "1" ] && tag="C${c}_nospec"
  echo "=== TOPC ${TC_TAG} $tag dp=$TC_DP batch=$TC_BATCH K=$TC_K $(date +%T) ==="
  timeout 6000 "$PY" "$SCRIPT" \
    > "$LOGD/topc_${TC_TAG}_${tag}.log" 2>&1
  echo "EXIT=$? ${TC_TAG} $tag"
  kill_stragglers
}

kill_stragglers
# Losslessness references (nospec + full-draft C=top_k) with token capture are
# run separately via run_lossless.sh; here we sweep accept/compute.
for c in $CLIST; do
  run_c "$c" 0
done
echo "TOPC SWEEP DONE ${TC_TAG} $(date +%T)"
