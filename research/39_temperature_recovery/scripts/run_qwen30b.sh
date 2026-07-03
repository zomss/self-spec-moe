#!/usr/bin/env bash
# Phase 39: Qwen3-30B-A3B headline -- temperature recovery + wall-clock speedup.
# DP=8 EP=8, FP8 full-replica comm-free draft, real forced-PCIe. Serial, one
# engine at a time, own-worker teardown. Runs:
#   nospec @ temp {0, 0.7, 1.0}   (the speedup denominators)
#   spec   @ temp {0, 0.7, 1.0}   FULL_CG (named headline)
#   spec   @ temp {0, 0.7, 1.0}   PIECEWISE (FULL_CG off; correct FA3 attention)
set -u
cd /data/smcho/self-spec-moe

PY=/data/smcho/self-spec-moe/.venv/bin/python
SCRIPT=research/39_temperature_recovery/scripts/qwen30b_temp.py
LOGDIR=research/39_temperature_recovery/logs
mkdir -p "$LOGDIR"

export NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=1
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
export VLLM_USE_V1=1

export W7_MODEL=/home/smcho/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39
export W7_DP=8 W7_TP=1 W7_DRAFT_QUANT=fp8
export W7_OUTLEN=160 W7_SHORTLEN=32 W7_ITERS=3 W7_WARMUP=2
export W7_GPU_MEM=${W7_GPU_MEM:-0.90} W7_MAX_MODEL_LEN=2048
export W7_BATCHES=${W7_BATCHES:-32} W7_KS=${W7_KS:-4}
export W7_LOG_A2A=1   # confirm draft real all-to-all = 0

TEMPS=${TEMPS:-"0.0 0.7 1.0"}

teardown() {
  pkill -9 -u "$(id -u)" -f "qwen30b_temp.py" 2>/dev/null
  pkill -9 -u "$(id -u)" -f "VLLM::EngineCore" 2>/dev/null
  pkill -9 -u "$(id -u)" -f "from multiprocessing.spawn" 2>/dev/null
  sleep 8
}

run_one() {  # mode temp full_cg tag
  local mode="$1" temp="$2" fcg="$3" tag="$4"
  local name="q30_${tag}_${mode}_t${temp//./p}_$([ "$fcg" = 1 ] && echo fcg || echo pw)"
  echo "=== RUN $name ($(date +%H:%M:%S)) ==="
  W7_TEMP="$temp" W7_FULL_CG="$fcg" W7_TAG="$tag" \
    timeout 5200 "$PY" "$SCRIPT" "$mode" >"$LOGDIR/${name}.log" 2>&1
  local rc=$?
  grep -h "^\[CFG\]\|^\[Q30T\]\|real=0\|a2a" "$LOGDIR/${name}.log" | tail -6 \
    || echo "  (no Q30T line; rc=$rc)"
  teardown
}

teardown

for t in $TEMPS; do
  run_one nospec "$t" 0 sweep
done
for t in $TEMPS; do
  run_one spec "$t" 1 sweep   # FULL_CG headline
done
for t in $TEMPS; do
  run_one spec "$t" 0 sweep   # PIECEWISE (correct FA3 attention)
done

echo "=== QWEN30B TEMP DONE ($(date +%H:%M:%S)) ==="
