#!/bin/bash
# Phase 97 P1: four non-scored static proxy boots on GPUs 4-5.
set -uo pipefail

REPO=/data/smcho/self-spec-moe
PHASE="$REPO/research/97_composition_runtime"
PYTHON="$REPO/.venv/bin/python"
TIMEOUT_S="${P97_BOOT_TIMEOUT_S:-1800}"

export PATH="$REPO/.venv/bin:$PATH"
export HF_HOME=/data/smcho/huggingface
export VLLM_CACHE_ROOT=/data/smcho/.cache/vllm
export TRITON_CACHE_DIR=/data/smcho/.cache/triton
export TMPDIR=/data/smcho/tmp
export VLLM_USE_FLASHINFER_SAMPLER=0
export LD_LIBRARY_PATH="/usr/local/cuda-12/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

mkdir -p "$PHASE/data/boot_proxy" "$PHASE/logs/boot_proxy"

gpu_used_mib() {
  nvidia-smi -i "$1" --query-gpu=memory.used \
    --format=csv,noheader,nounits 2>/dev/null
}

wait_free() {
  local gpu=$1 used attempt
  for attempt in $(seq 1 60); do
    used=$(gpu_used_mib "$gpu")
    if [ "${used:-999999}" -lt 2000 ]; then
      return 0
    fi
    sleep 2
  done
  echo "[P97] GPU $gpu still uses ${used:-unknown} MiB; refusing next class"
  return 1
}

run_one() {
  local gpu=$1 boot_class=$2
  local out="$PHASE/data/boot_proxy/$boot_class.json"
  local log="$PHASE/logs/boot_proxy/$boot_class.log"
  local used
  used=$(gpu_used_mib "$gpu")
  if [ "${used:-999999}" -ge 2000 ]; then
    echo "[P97] refuse $boot_class: GPU $gpu already uses ${used:-unknown} MiB"
    return 1
  fi
  echo "[P97] start $boot_class on GPU $gpu"
  if CUDA_VISIBLE_DEVICES="$gpu" P97_GPU_PHYSICAL="$gpu" \
      timeout --signal=TERM --kill-after=60s "${TIMEOUT_S}s" \
      "$PYTHON" "$PHASE/scripts/inventory_boot.py" \
      --boot-class "$boot_class" --gpu "$gpu" --out "$out" \
      >"$log" 2>&1; then
    echo "[P97] done $boot_class on GPU $gpu"
  else
    local status=$?
    echo "[P97] failed $boot_class on GPU $gpu (status $status; $log)"
    tail -n 20 "$log"
    wait_free "$gpu" || true
    return "$status"
  fi
  wait_free "$gpu"
}

run_lane() {
  local gpu=$1 first=$2 second=$3 lane_status=0
  run_one "$gpu" "$first" || lane_status=1
  if wait_free "$gpu"; then
    run_one "$gpu" "$second" || lane_status=1
  else
    lane_status=1
  fi
  return "$lane_status"
}

cd "$REPO" || exit 1
run_lane 4 baseline weight_q &
pid4=$!
run_lane 5 kv_q weight_q_kv_q &
pid5=$!

matrix_status=0
wait "$pid4" || matrix_status=1
wait "$pid5" || matrix_status=1

if ! "$PYTHON" "$PHASE/scripts/summarize_boots.py" \
    --input-dir "$PHASE/data/boot_proxy" \
    --out "$PHASE/data/boot_resource_ledger.json"; then
  matrix_status=1
fi
exit "$matrix_status"
