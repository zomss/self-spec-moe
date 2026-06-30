#!/usr/bin/env bash
# Phase 40 Step 2/3: accept_len for A/C x {bf16-accum, fp32-accum} at DP=8 K=4.
# Self-spec draft_model, Qwen1.5-MoE, forced-PCIe, bf16, FULL_CG + compile-
# consistent (the only remaining divergence is comm-free-vs-EP). The fp32-accum
# fix should pull config A's accept_len up toward config C's ~5.0.
# Tears down OWN workers between runs (idle GPUs only; never others').
set -u
cd /data/smcho/self-spec-moe

export PYTHONPATH=/data/smcho/ssm-num
export NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=1
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 VLLM_USE_V1=1

PY=/data/smcho/self-spec-moe/.venv/bin/python
SCRIPT=research/40_num_divergence/scripts/accept_run.py
LOGDIR=research/40_num_divergence/logs
mkdir -p "$LOGDIR"

teardown() {
  pkill -9 -u "$(id -u)" -f "accept_run.py" 2>/dev/null
  pkill -9 -u "$(id -u)" -f "VLLM::EngineCore" 2>/dev/null
  pkill -9 -u "$(id -u)" -f "from multiprocessing.spawn" 2>/dev/null
  sleep 6
  return 0
}

run_one() {
  local cfg="$1"; local fp32="$2"; local k="${3:-4}"; local dp="${4:-8}"
  local name="acc_${cfg}_dp${dp}_K${k}_cg$([ "$fp32" = 1 ] && echo _fp32 || echo)"
  echo "=== RUN $name ($(date +%H:%M:%S)) ==="
  CFG="$cfg" E_FP32="$fp32" E_K="$k" E_DP="$dp" E_TAG="acc" \
    timeout 1800 "$PY" "$SCRIPT" >"$LOGDIR/${name}.log" 2>&1
  local rc=$?
  grep -h "^RESULT" "$LOGDIR/${name}.log" || echo "  (no RESULT; rc=$rc)"
  teardown
}

teardown  # clean slate

# Step 2: accept_len before/after the fp32-accum fix, at DP=8 K=4.
run_one A 0   # config A, bf16-accum (the collapse: ~2.18)
run_one A 1   # config A, fp32-accum (the fix: should rise toward ~5.0)
run_one C 0   # config C, bf16-accum (upper bound ~5.0)
run_one C 1   # config C, fp32-accum (should stay ~5.0)

echo "=== ALL DONE ($(date +%H:%M:%S)) ==="
