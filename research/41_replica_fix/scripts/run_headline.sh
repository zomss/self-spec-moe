#!/usr/bin/env bash
# Phase 41 HEADLINE: Qwen3-30B-A3B, DP=8/EP=8, FP8 full-replica comm-free draft
# vs no-spec. Forced-PCIe, FULL_CG + COMPILE_CONSISTENT. Two-length-slope decode
# tok/s (w7_fp8_timing.py). Serial: one engine at a time, kill stragglers between.
set -u
cd /data/smcho/ssm-num

export PYTHONPATH=/data/smcho/ssm-num
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
export NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=1
# FULL draft cudagraphs + batch-invariant numerics (the W7 fixed config).
export VLLM_SELF_SPEC_DRAFT_FULL_CG=1 VLLM_SELF_SPEC_COMPILE_CONSISTENT=1

export W7_MODEL=/home/smcho/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39
export W7_DP=8 W7_TP=1 W7_TRC=0
export W7_OUT=/data/smcho/ssm-num/research/41_replica_fix/data
export W7_EAGER=0                 # CUDA graphs ON (+ FULL_CG via env above)
export W7_BATCHES="${W7_BATCHES:-16,64,128}"
export W7_KS="${W7_KS:-3}"
export W7_OUTLEN="${W7_OUTLEN:-160}" W7_SHORTLEN="${W7_SHORTLEN:-32}"
export W7_ITERS="${W7_ITERS:-3}" W7_WARMUP="${W7_WARMUP:-2}"
export W7_GPU_MEM="${W7_GPU_MEM:-0.90}"

PY=/data/smcho/self-spec-moe/.venv/bin/python
SCRIPT=research/34_worldA_system/scripts/w7_fp8_timing.py
LOGD=/data/smcho/ssm-num/research/41_replica_fix/logs

kill_stragglers() {
  pkill -9 -f "w7_fp8_timing" 2>/dev/null
  for pid in $(ps -u "$USER" -o pid,cmd | grep -E "EngineCore" | grep -v grep | awk '{print $1}'); do
    kill -9 "$pid" 2>/dev/null
  done
  sleep 4
}

run() {  # mode tag draft_quant
  local mode="$1" tag="$2" dq="${3:-}"
  echo "=== HEADLINE mode=$mode tag=$tag draft_quant=${dq:-bf16} dp=$W7_DP $(date +%T) ==="
  W7_TAG="$tag" W7_DRAFT_QUANT="$dq" timeout 5000 "$PY" "$SCRIPT" "$mode" \
    > "$LOGD/headline_${tag}_${mode}.log" 2>&1
  echo "EXIT=$? mode=$mode"
  kill_stragglers
}

kill_stragglers
run nospec qwen30b_h           # bf16 target, no spec
run spec   qwen30b_h_fp8  fp8  # FP8 full-replica comm-free draft
echo "HEADLINE DONE $(date +%T)"
