#!/usr/bin/env bash
# Phase 39: proxy (Qwen1.5-MoE-A2.7B) temperature sweep. Serial, one DP=8 group
# at a time. Full-replica comm-free draft + lossless standard rejection sampler
# + probabilistic draft. Tears down OWN vLLM workers between runs (idle GPUs
# only; never kill others').
set -u
cd /data/smcho/self-spec-moe

PY=/data/smcho/self-spec-moe/.venv/bin/python
SCRIPT=research/39_temperature_recovery/scripts/temp_sweep.py
LOGDIR=research/39_temperature_recovery/logs
mkdir -p "$LOGDIR"

export NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=1
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
export VLLM_USE_V1=1

teardown() {
  pkill -9 -u "$(id -u)" -f "temp_sweep.py" 2>/dev/null
  pkill -9 -u "$(id -u)" -f "VLLM::EngineCore" 2>/dev/null
  pkill -9 -u "$(id -u)" -f "from multiprocessing.spawn" 2>/dev/null
  sleep 6
}

run_one() {
  local temp="$1"; local prob="$2"; local tag="$3"
  local name="${tag}_t${temp}_$([ "$prob" = 1 ] && echo prob || echo greedydraft)_dp8_K4"
  name=${name//./p}
  echo "=== RUN $name ($(date +%H:%M:%S)) ==="
  E_TEMP="$temp" E_PROBABILISTIC="$prob" E_DP=8 E_K=4 E_TAG="$tag" \
    timeout 2600 "$PY" "$SCRIPT" >"$LOGDIR/${name}.log" 2>&1
  local rc=$?
  grep -h "^\[CFG\]" "$LOGDIR/${name}.log" || true
  grep -h "^RESULT" "$LOGDIR/${name}.log" || echo "  (no RESULT line; rc=$rc)"
  teardown
}

teardown  # clean slate

# The test: temperature sweep at DP=8, full-replica comm-free draft, probabilistic.
run_one 0.0 1 sweep   # greedy baseline cross-check (~2.18 phase 38)
run_one 0.5 1 sweep
run_one 0.7 1 sweep
run_one 1.0 1 sweep

echo "=== PROXY SWEEP DONE ($(date +%H:%M:%S)) ==="
