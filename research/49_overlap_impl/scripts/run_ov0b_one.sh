#!/usr/bin/env bash
# Single OV0b probe run on GPUs 0-3 (DP4). Usage:
#   TAG=<tag> A2A=<us> SHADOW=<n> [EXTRA_ENV="K=V K=V"] bash run_ov0b_one.sh
set -u
REPO=/data/smcho/self-spec-moe
cd "$REPO"
export PYTHONPATH="$REPO"
export PATH="$REPO/.venv/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OV0B_MARK=1
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
export NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=1
export VLLM_SELF_SPEC_DRAFT_FULL_REPLICA=1 VLLM_SELF_SPEC_DRAFT_LOCAL_ROUTE=1
export VLLM_SELF_SPEC_DRAFT_FULL_CG=1 VLLM_SELF_SPEC_COMPILE_CONSISTENT=1
export VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1
export VLLM_SELF_SPEC_DRAFT_GRAPH_POOL=1
# Dedicated MoE-scratch workspace for the draft (OV0b root-cause fix); override
# with EXTRA_ENV="VLLM_SELF_SPEC_DRAFT_WORKSPACE=0" for the A/B.
export VLLM_SELF_SPEC_DRAFT_WORKSPACE=1
export W7_MODEL=/home/smcho/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39
export W7_DP=4 W7_TP=1 W7_TRC=0 W7_EAGER=0 W7_ITERS=3 W7_WARMUP=2
export W7_KS=2 W7_BATCHES=64
export W7_OUT="$REPO/research/49_overlap_impl/data"
export VLLM_SELF_SPEC_SHADOW_CHAIN="${SHADOW:-2}"
export W7_TAG="${TAG:?}" W7_A2A_US="${A2A:-0}" W7_DRAFT_QUANT=fp8
for kv in ${EXTRA_ENV:-}; do export "$kv"; done

LOGD="$REPO/research/49_overlap_impl/logs"
mkdir -p "$LOGD"
# Pre-run settle: back-to-back launches raced the previous run's dying workers
# for the DP master port (EADDRINUSE -> hung engines). Wait until no marked
# process remains, then give the kernel time to release sockets.
for _ in $(seq 1 60); do
  alive=0
  for pid in $(ps -u "$USER" -o pid,cmd | grep -E "w7_fp8_timing|EngineCore|VLLM::" | grep -v grep | awk '{print $1}'); do
    tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | grep -q "^OV0B_MARK=1$" && alive=1
  done
  [ "$alive" -eq 0 ] && break
  sleep 5
done
sleep 20

cleanup() {
  pkill -9 -f "w7_fp8_timing" 2>/dev/null
  # Marker-based kill (the OV0B_MARK env does not survive every worker spawn
  # path, so this alone leaks workers)...
  for pid in $(ps -u "$USER" -o pid,cmd | grep -E "EngineCore|VLLM::" | grep -v grep | awk '{print $1}'); do
    tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | grep -q "^OV0B_MARK=1$" && kill -9 "$pid" 2>/dev/null
  done
  # ...so ALSO kill any of OUR user's VLLM processes holding memory on OUR
  # GPUs (0-3). Never touches other users (kill fails cross-user) nor GPUs
  # 4-7 (bus-id scoped).
  local buses
  buses=$(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader \
    | awk -F', ' '$1 <= 3 {print $2}')
  for line in $(nvidia-smi --query-compute-apps=gpu_bus_id,pid --format=csv,noheader | tr -d ' '); do
    bus="${line%,*}"; pid="${line#*,}"
    echo "$buses" | grep -q "$bus" || continue
    ps -o user= -p "$pid" 2>/dev/null | grep -q "^$USER$" || continue
    ps -o cmd= -p "$pid" 2>/dev/null | grep -q "VLLM\|w7_fp8" && kill -9 "$pid" 2>/dev/null
  done
  sleep 5
}

# Engine boot flakes on this shared box (EADDRINUSE races against other
# users' engines grabbing ports + TIME_WAIT from our own teardown; a failed
# boot can then wedge a worker for 600 s on the dead port). Bound the boot
# damage with a shorter timeout on failure detection and retry up to 3x.
for attempt in 1 2 3; do
  timeout 6000 "$REPO/.venv/bin/python" \
    "$REPO/research/34_worldA_system/scripts/w7_fp8_timing.py" spec \
    > "$LOGD/${TAG}.log" 2>&1 &
  HARNESS_PID=$!
  # Watch for a failed boot: kill the harness early instead of letting a
  # worker wait 600 s on a dead port.
  while kill -0 "$HARNESS_PID" 2>/dev/null; do
    if grep -q "Engine core initialization failed" "$LOGD/${TAG}.log" 2>/dev/null; then
      echo "BOOT FAILED (attempt $attempt) -- killing and retrying"
      kill -9 "$HARNESS_PID" 2>/dev/null
      break
    fi
    sleep 10
  done
  wait "$HARNESS_PID" 2>/dev/null
  RC=$?
  cleanup
  if ! grep -q "Engine core initialization failed" "$LOGD/${TAG}.log" 2>/dev/null; then
    echo "EXIT=$RC attempt=$attempt"
    break
  fi
  sleep 60
done
true
