#!/usr/bin/env bash
# Phase 39: losslessness spot-check under temperature (proxy DP=8). Runs nospec
# then spec at temp 0.7 fixed seed, one engine at a time, own-worker teardown.
set -u
cd /data/smcho/self-spec-moe

PY=/data/smcho/self-spec-moe/.venv/bin/python
SCRIPT=research/39_temperature_recovery/scripts/lossless.py
LOGDIR=research/39_temperature_recovery/logs
mkdir -p "$LOGDIR"

export NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=1
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
export VLLM_USE_V1=1
export E_TEMP=${E_TEMP:-0.7} E_SEED=${E_SEED:-0} E_DP=8 E_K=4

teardown() {
  pkill -9 -u "$(id -u)" -f "lossless.py" 2>/dev/null
  pkill -9 -u "$(id -u)" -f "VLLM::EngineCore" 2>/dev/null
  pkill -9 -u "$(id -u)" -f "from multiprocessing.spawn" 2>/dev/null
  sleep 6
}

run_one() {
  local mode="$1"
  echo "=== RUN lossless_${mode}_t${E_TEMP} ($(date +%H:%M:%S)) ==="
  timeout 2600 "$PY" "$SCRIPT" "$mode" \
    >"$LOGDIR/lossless_${mode}_t${E_TEMP//./p}.log" 2>&1
  grep -h "^\[CFG\]\|^\[LL\]" "$LOGDIR/lossless_${mode}_t${E_TEMP//./p}.log" || \
    echo "  (no LL line)"
  teardown
}

teardown
run_one nospec
run_one spec
echo "=== LOSSLESS DONE ($(date +%H:%M:%S)) ==="
