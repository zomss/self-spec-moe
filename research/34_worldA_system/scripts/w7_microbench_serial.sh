#!/usr/bin/env bash
# Serial driver for the W7 micro-bench. Spec engines must run ONE AT A TIME
# (the draft loop is CPU-orchestration-bound; a concurrent engine inflates step
# time ~7x -- see results_W7.md). All runs forced-PCIe + emulated A2A.
set -u

PY=/data/smcho/self-spec-moe/.venv/bin/python
SCRIPT=/data/smcho/ssm-mb/research/34_worldA_system/scripts/w7_microbench.py
LOGDIR=/data/smcho/ssm-mb/research/34_worldA_system/logs
mkdir -p "$LOGDIR"

export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH=/data/smcho/ssm-mb
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
export NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=1

run() {
  local mode="$1"; local tag="$2"; local batch="$3"; local k="$4"
  shift 4
  local label="${mode}_${tag}_b${batch}_K${k}"
  echo "=== RUN $label ($(date +%T)) ==="
  env MB_MODE="$mode" MB_TAG="$tag" MB_BATCH="$batch" MB_K="$k" \
      MB_A2A_US=100 MB_STEADY=100 MB_WARMUP=50 "$@" \
      "$PY" "$SCRIPT" "$mode" > "$LOGDIR/mb_${label}.log" 2>&1
  echo "    exit=$? ; $(grep -h '^\[MB\] ' "$LOGDIR/mb_${label}.log" | tail -8 | tr '\n' '|')"
}

# --- DeepSeek-V2-Lite (native deepseek_v2) ---
# nospec baselines (t_nospec_step) at the batches we compare.
run nospec v2lite 64 0
run nospec v2lite 8  0
# Primary spec point: b64 K2 (the 0.42x point).
run spec   v2lite 64 2
# Secondary contrasts.
run spec   v2lite 64 4
run spec   v2lite 8  2

# --- Qwen3-30B-A3B (bigger forward) one point if MB_RUN_QWEN=1 ---
if [ "${MB_RUN_QWEN:-0}" = "1" ]; then
  QWEN=Qwen/Qwen3-30B-A3B
  run nospec qwen30b 64 0 MB_MODEL="$QWEN" MB_TRC=1
  run spec   qwen30b 64 2 MB_MODEL="$QWEN" MB_TRC=1
fi

echo "=== ALL DONE ($(date +%T)) ==="
