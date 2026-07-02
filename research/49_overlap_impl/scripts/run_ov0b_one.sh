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
timeout 6000 "$REPO/.venv/bin/python" \
  "$REPO/research/34_worldA_system/scripts/w7_fp8_timing.py" spec \
  > "$LOGD/${TAG}.log" 2>&1
echo "EXIT=$?"
pkill -9 -f "w7_fp8_timing" 2>/dev/null
for pid in $(ps -u "$USER" -o pid,cmd | grep -E "EngineCore|VLLM::" | grep -v grep | awk '{print $1}'); do
  tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | grep -q "^OV0B_MARK=1$" && kill -9 "$pid" 2>/dev/null
done
true
