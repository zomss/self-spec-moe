#!/usr/bin/env bash
# Phase 44 RECON: decompose the comm-free self-spec cycle and reconcile it
# against the C1 cost model. Qwen3-30B-A3B, DP=8/EP=8, forced-PCIe, FP8
# full-replica comm-free draft, K=2, batch 64, a2a=0 (native forced-PCIe).
#
# Runs three engines SERIALLY (one DP=8 group at a time), tearing down our OWN
# workers between runs:
#   spec     -> real cycle decomposition + full-replica draft forward
#   nospec   -> verify/no-spec compute reference (~34ms)
#   skiptile -> the T_compute skip-A2A tile draft forward at batch 64
#
# Config mirrors research/43_kretune/scripts/run_kretune.sh (the 0.55x/1021 tok/s
# operating point). NO vllm/ changes.
set -u
REPO=/data/smcho/self-spec-moe
cd "$REPO"

export PYTHONPATH="$REPO"
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
export NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=1
export VLLM_SELF_SPEC_DRAFT_FULL_CG=1 VLLM_SELF_SPEC_COMPILE_CONSISTENT=1
export VLLM_SELF_SPEC_LOG_A2A_COUNTS=1

export RC_MODEL=/home/smcho/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39
export RC_DP=8 RC_TP=1 RC_TRC=0
export RC_OUT="$REPO/research/44_recon/data"
export RC_BATCH="${RC_BATCH:-64}" RC_K="${RC_K:-2}"
export RC_DRAFT_QUANT="${RC_DRAFT_QUANT:-fp8}"
export RC_GPU_MEM="${RC_GPU_MEM:-0.90}"
export RC_STEADY="${RC_STEADY:-80}" RC_WARMUP="${RC_WARMUP:-60}"
export RC_TAG="${RC_TAG:-qwen30b}"

PY="$REPO/.venv/bin/python"
SCRIPT="$REPO/research/44_recon/scripts/recon_microbench.py"
LOGD="$REPO/research/44_recon/logs"
mkdir -p "$LOGD" "$RC_OUT"

kill_stragglers() {
  pkill -9 -f "recon_microbench" 2>/dev/null
  for pid in $(ps -u "$USER" -o pid,cmd | grep -E "EngineCore|VLLM::" | grep -v grep | awk '{print $1}'); do
    kill -9 "$pid" 2>/dev/null
  done
  sleep 6
}

run_mode() {  # mode
  local mode="$1"
  echo "=== RECON mode=$mode dp=$RC_DP batch=$RC_BATCH K=$RC_K $(date +%T) ==="
  timeout 6000 "$PY" "$SCRIPT" "$mode" \
    > "$LOGD/recon_${mode}_b${RC_BATCH}_K${RC_K}.log" 2>&1
  echo "EXIT=$? mode=$mode"
  kill_stragglers
}

MODES="${MODES:-spec nospec skiptile}"
kill_stragglers
for m in $MODES; do
  run_mode "$m"
done
echo "RECON DONE $(date +%T)"
