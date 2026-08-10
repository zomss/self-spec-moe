#!/bin/bash
# Phase 97: registered MBT2048/4096 x compiled/scoped-eager diagnostics.
set -uo pipefail

REPO=/data/smcho/self-spec-moe
PHASE="$REPO/research/97_composition_runtime"
PYTHON="$REPO/.venv/bin/python"
TIMEOUT_S="${P97_WARMUP_TIMEOUT_S:-900}"

export PATH="$REPO/.venv/bin:$PATH"
export HF_HOME=/data/smcho/huggingface
export TRITON_CACHE_DIR=/data/smcho/.cache/triton
export TMPDIR=/data/smcho/tmp
export VLLM_USE_FLASHINFER_SAMPLER=0
export LD_LIBRARY_PATH="/usr/local/cuda-12/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

mkdir -p "$PHASE/data/diagnostics" "$PHASE/logs/diagnostics"

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
  echo "[P97] GPU $gpu still uses ${used:-unknown} MiB"
  return 1
}

run_one() {
  local gpu=$1 mbt=$2 mode=$3
  local stem="weight_q_kv_q_mbt${mbt}_${mode}"
  local out="$PHASE/data/diagnostics/$stem.json"
  local log="$PHASE/logs/diagnostics/$stem.log"
  local cache_root="/data/smcho/.cache/vllm-p97-warmup/$stem"
  local used status

  if [ -e "$out" ] || [ -e "$log" ]; then
    echo "[P97] refuse overwrite of existing diagnostic $stem"
    return 2
  fi
  used=$(gpu_used_mib "$gpu")
  if [ "${used:-999999}" -ge 2000 ]; then
    echo "[P97] refuse $stem: GPU $gpu already uses ${used:-unknown} MiB"
    return 2
  fi

  mkdir -p "$cache_root"
  echo "[P97] start $stem on GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" P97_GPU_PHYSICAL="$gpu" \
    VLLM_CACHE_ROOT="$cache_root" \
    timeout --signal=TERM --kill-after=60s "${TIMEOUT_S}s" \
    "$PYTHON" "$PHASE/scripts/inventory_boot.py" \
    --boot-class weight_q_kv_q --gpu "$gpu" --out "$out" \
    --max-num-batched-tokens "$mbt" --draft-execution "$mode" \
    >"$log" 2>&1
  status=$?

  if [ "$status" -eq 0 ]; then
    echo "[P97] done $stem on GPU $gpu"
  else
    echo "[P97] terminal status $status for $stem"
    tail -n 20 "$log"
  fi

  wait_free "$gpu" || return 3
  if [ ! -s "$out" ]; then
    "$PYTHON" "$PHASE/scripts/record_warmup_timeout.py" \
      --out "$out" --log "$log" --gpu "$gpu" --mbt "$mbt" \
      --draft-execution "$mode" --timeout-seconds "$TIMEOUT_S" \
      --runner-status "$status"
  fi
  return 0
}

run_pair() {
  local mbt=$1 compiled_gpu=$2 eager_gpu=$3
  local pair_status=0 pid_compiled pid_eager
  run_one "$compiled_gpu" "$mbt" compiled &
  pid_compiled=$!
  run_one "$eager_gpu" "$mbt" eager &
  pid_eager=$!
  wait "$pid_compiled" || pair_status=1
  wait "$pid_eager" || pair_status=1
  return "$pair_status"
}

cd "$REPO" || exit 1
matrix_status=0
run_pair 2048 4 5 || matrix_status=1
run_pair 4096 5 4 || matrix_status=1

"$PYTHON" "$PHASE/scripts/summarize_warmup_diagnostics.py" \
  --phase "$PHASE" --out "$PHASE/data/warmup_diagnostic_ledger.json" \
  || matrix_status=1

"$PYTHON" "$PHASE/scripts/summarize_boots.py" \
  --input-dir "$PHASE/data/boot_proxy" \
  --out "$PHASE/data/boot_resource_ledger.json" >/dev/null 2>&1 || true

exit "$matrix_status"
