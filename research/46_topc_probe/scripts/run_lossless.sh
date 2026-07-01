#!/usr/bin/env bash
# Phase 46 losslessness sanity: at ONE C, confirm the spec (top-C draft) output
# token ids are byte-identical to the no-spec output (the full top_k verify makes
# it lossless regardless of C). Runs two engines serially with token capture,
# then diffs.
#   run_lossless.sh <model_key> <C>
set -u
REPO=/data/smcho/ssm-topc
cd "$REPO"

export PYTHONPATH="$REPO"
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
export NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=1
export VLLM_SELF_SPEC_DRAFT_FULL_REPLICA=1
export VLLM_SELF_SPEC_DRAFT_LOCAL_ROUTE=1
export VLLM_SELF_SPEC_DRAFT_FULL_CG=1
export VLLM_SELF_SPEC_COMPILE_CONSISTENT=1
export VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1

MODEL_KEY="${1:?model_key required}"
CVAL="${2:?C required}"

case "$MODEL_KEY" in
  qwen30b)
    export TC_MODEL=/home/smcho/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39
    export TC_TAG=qwen30b ;;
  qwen15moe)
    export TC_MODEL=/home/smcho/.cache/huggingface/hub/models--Qwen--Qwen1.5-MoE-A2.7B/snapshots/1a758c50ecb6350748b9ce0a99d2352fd9fc11c9
    export TC_TAG=qwen15moe ;;
  *) echo "unknown model_key $MODEL_KEY"; exit 1;;
esac

export TC_TRC=0 TC_DP="${TC_DP:-8}" TC_TP=1
export TC_OUT="$REPO/research/46_topc_probe/data"
export TC_BATCH="${TC_BATCH:-64}" TC_K="${TC_K:-2}"
export TC_DRAFT_QUANT="${TC_DRAFT_QUANT:-fp8}"
export TC_GPU_MEM="${TC_GPU_MEM:-0.90}"
export TC_STEADY="${TC_STEADY:-40}" TC_WARMUP="${TC_WARMUP:-20}"
export TC_CAPTURE_TOKENS=1

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

kill_stragglers
# no-spec reference (full model, no draft)
export TC_C="$CVAL" TC_NOSPEC=1
echo "=== LOSSLESS ${TC_TAG} nospec $(date +%T) ==="
timeout 3000 "$PY" "$SCRIPT" > "$LOGD/lossless_${TC_TAG}_nospec.log" 2>&1
echo "EXIT=$? nospec"
kill_stragglers
# spec with top-C draft
export TC_NOSPEC=0
echo "=== LOSSLESS ${TC_TAG} spec C=$CVAL $(date +%T) ==="
timeout 3000 "$PY" "$SCRIPT" > "$LOGD/lossless_${TC_TAG}_C${CVAL}.log" 2>&1
echo "EXIT=$? spec C=$CVAL"
kill_stragglers

BASE="$TC_OUT/tc_${TC_TAG}_b${TC_BATCH}_K${TC_K}_C${CVAL}_nospec.json"
SPEC="$TC_OUT/tc_${TC_TAG}_b${TC_BATCH}_K${TC_K}_C${CVAL}.json"
echo "=== DIFF ==="
"$PY" "$REPO/research/46_topc_probe/scripts/diff_tokens.py" "$BASE" "$SPEC"
echo "LOSSLESS DONE ${TC_TAG} $(date +%T)"
