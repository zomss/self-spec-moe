#!/bin/bash
# Phase 97: registered Cutlass isolation for W4A8/private-FP8 MBT2048.
set -uo pipefail

REPO=/data/smcho/self-spec-moe
PHASE="$REPO/research/97_composition_runtime"
PYTHON="$REPO/.venv/bin/python"
GPU=5
MBT=2048
MODE="${P97_CUTLASS_DRAFT_EXECUTION:-eager}"
TIMEOUT_S="${P97_CUTLASS_TIMEOUT_S:-900}"
STEM="weight_q_kv_q_mbt${MBT}_${MODE}_cutlass"
OUT="$PHASE/data/diagnostics/$STEM.json"
LOG="$PHASE/logs/diagnostics/$STEM.log"
GPU_LOG="$PHASE/logs/diagnostics/$STEM.gpu.csv"
CACHE_ROOT="/data/smcho/.cache/vllm-p97-warmup/$STEM"
sampler_pid=

export PATH="$REPO/.venv/bin:$PATH"
export HF_HOME=/data/smcho/huggingface
export TRITON_CACHE_DIR=/data/smcho/.cache/triton
export TMPDIR=/data/smcho/tmp
export VLLM_USE_FLASHINFER_SAMPLER=0
export LD_LIBRARY_PATH="/usr/local/cuda-12/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

mkdir -p "$PHASE/data/diagnostics" "$PHASE/logs/diagnostics" "$CACHE_ROOT"

gpu_used_mib() {
  nvidia-smi -i "$1" --query-gpu=memory.used \
    --format=csv,noheader,nounits 2>/dev/null
}

wait_free() {
  local used attempt
  for attempt in $(seq 1 60); do
    used=$(gpu_used_mib "$GPU")
    if [ "${used:-999999}" -lt 2000 ]; then
      return 0
    fi
    sleep 2
  done
  echo "[P97] GPU $GPU still uses ${used:-unknown} MiB"
  return 1
}

cleanup_sampler() {
  if [ -n "${sampler_pid:-}" ]; then
    kill "$sampler_pid" 2>/dev/null || true
    wait "$sampler_pid" 2>/dev/null || true
    sampler_pid=
  fi
}

trap cleanup_sampler EXIT INT TERM

if [ "$MODE" != eager ] && [ "$MODE" != compiled ]; then
  echo "[P97] unsupported draft execution: $MODE"
  exit 2
fi
if [ -e "$OUT" ] || [ -e "$LOG" ] || [ -e "$GPU_LOG" ]; then
  echo "[P97] refuse overwrite of existing diagnostic $STEM"
  exit 2
fi

used=$(gpu_used_mib "$GPU")
if [ "${used:-999999}" -ge 2000 ]; then
  echo "[P97] refuse $STEM: GPU $GPU already uses ${used:-unknown} MiB"
  exit 2
fi

nvidia-smi -i "$GPU" \
  --query-gpu=timestamp,memory.used,utilization.gpu,utilization.memory,power.draw \
  --format=csv,noheader,nounits --loop-ms=500 >"$GPU_LOG" &
sampler_pid=$!

echo "[P97] start $STEM on GPU $GPU"
CUDA_VISIBLE_DEVICES="$GPU" P97_GPU_PHYSICAL="$GPU" \
  VLLM_CACHE_ROOT="$CACHE_ROOT" \
  timeout --signal=TERM --kill-after=60s "${TIMEOUT_S}s" \
  "$PYTHON" "$PHASE/scripts/inventory_boot.py" \
  --boot-class diagnostic_weight_q_kv_q_cutlass \
  --gpu "$GPU" --out "$OUT" --max-num-batched-tokens "$MBT" \
  --draft-execution "$MODE" >"$LOG" 2>&1
status=$?
cleanup_sampler

if [ "$status" -eq 0 ]; then
  echo "[P97] done $STEM on GPU $GPU"
else
  echo "[P97] terminal status $status for $STEM"
  tail -n 40 "$LOG"
fi

wait_free || exit 3
if [ ! -s "$OUT" ]; then
  "$PYTHON" "$PHASE/scripts/record_warmup_timeout.py" \
    --out "$OUT" --log "$LOG" --gpu "$GPU" --mbt "$MBT" \
    --draft-execution "$MODE" --timeout-seconds "$TIMEOUT_S" \
    --runner-status "$status" \
    --boot-class diagnostic_weight_q_kv_q_cutlass \
    --draft-kv-dtype fp8 --linear-backend cutlass
fi

if ! rg -q \
  "Using CutlassW4A8LinearKernel for CompressedTensorsW4A8Fp8" "$LOG"; then
  echo "[P97] fail closed: Cutlass W4A8 selection evidence is missing"
  exit 4
fi

if [ "$status" -eq 0 ]; then
  exit 0
fi
exit 1
