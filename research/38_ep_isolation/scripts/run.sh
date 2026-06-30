#!/usr/bin/env bash
# Phase 38: decisive EP-isolation runner. Serial, one DP=8 group at a time.
# Tears down OWN vLLM workers between runs (idle GPUs only; never kill others').
set -u
cd /data/smcho/self-spec-moe

PY=/data/smcho/self-spec-moe/.venv/bin/python
SCRIPT=research/38_ep_isolation/scripts/ep_isolation.py
LOGDIR=research/38_ep_isolation/logs
mkdir -p "$LOGDIR"

# forced-PCIe + no deep-gemm + our scratch HF cache for any temp quant
export NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=1
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
export VLLM_USE_V1=1

# Kill only OUR leaked EngineCore/vLLM workers (match this repo path), never others.
teardown() {
  pkill -9 -u "$(id -u)" -f "ep_isolation.py" 2>/dev/null
  pkill -9 -u "$(id -u)" -f "VLLM::EngineCore" 2>/dev/null
  pkill -9 -u "$(id -u)" -f "from multiprocessing.spawn" 2>/dev/null
  sleep 6
}

run_one() {
  local cfg="$1"; local k="$2"; local dp="$3"; local eager="$4"; local tag="$5"
  local name="${tag}_${cfg}_dp${dp}_K${k}$([ "$eager" = 1 ] && echo _eager || echo _cg)"
  echo "=== RUN $name ($(date +%H:%M:%S)) ==="
  CFG="$cfg" E_K="$k" E_DP="$dp" E_EAGER="$eager" E_TAG="$tag" \
    timeout 2600 "$PY" "$SCRIPT" >"$LOGDIR/${name}.log" 2>&1
  local rc=$?
  grep -h "^RESULT" "$LOGDIR/${name}.log" || echo "  (no RESULT line; rc=$rc)"
  teardown
}

teardown  # clean slate

# ---- Test 1: A/B/C at DP=8, K=4, CUDA graphs ON (FULL_CG + compile-consistent) ----
run_one A 4 8 0 t1
run_one B 4 8 0 t1
run_one C 4 8 0 t1

# ---- Test 2: config A (the collapse), per-position already captured at K=4 above.
# Add K=1 (depth-0 only) for A and C as a cross-check (per-step isolation). ----
run_one A 1 8 0 t2
run_one C 1 8 0 t2

echo "=== ALL DONE ($(date +%H:%M:%S)) ==="
