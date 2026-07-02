#!/usr/bin/env bash
# OV0b: in-engine shadow-chain overlap probe A/B.
# 4 runs: shadow {off, on(2 steps)} x a2a {0, 500} us/coll, batch 64, K=2, the
# Phase-48 fully-optimized stack (shielded delay gate). GO if at a2a=500
# tok/s(on) ~= tok/s(off) (chain hides in the comm window) AND accept_len
# unchanged (no cudagraph memory-pool aliasing between draft/verify graphs).
set -u
REPO=/data/smcho/self-spec-moe
cd "$REPO"

export PYTHONPATH="$REPO"
export PATH="$REPO/.venv/bin:$PATH"
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
export NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=1
export VLLM_SELF_SPEC_DRAFT_FULL_REPLICA=1
export VLLM_SELF_SPEC_DRAFT_LOCAL_ROUTE=1
export VLLM_SELF_SPEC_DRAFT_FULL_CG=1
export VLLM_SELF_SPEC_COMPILE_CONSISTENT=1
export VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1

export W7_MODEL=/home/smcho/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39
export W7_DP=8 W7_TP=1 W7_TRC=0
export W7_OUT="$REPO/research/49_overlap_impl/data"
export W7_EAGER=0
export W7_OUTLEN="${W7_OUTLEN:-160}" W7_SHORTLEN="${W7_SHORTLEN:-32}"
export W7_ITERS="${W7_ITERS:-3}" W7_WARMUP="${W7_WARMUP:-2}"
export W7_GPU_MEM="${W7_GPU_MEM:-0.90}"
export W7_KS=2
export W7_BATCHES=64

PY="$REPO/.venv/bin/python"
SCRIPT="$REPO/research/34_worldA_system/scripts/w7_fp8_timing.py"
LOGD="$REPO/research/49_overlap_impl/logs"
mkdir -p "$LOGD" "$W7_OUT"

kill_stragglers() {
  pkill -9 -f "w7_fp8_timing" 2>/dev/null
  for pid in $(ps -u "$USER" -o pid,cmd | grep -E "EngineCore|VLLM::" | grep -v grep | awk '{print $1}'); do
    kill -9 "$pid" 2>/dev/null
  done
  sleep 5
}

run() {  # tag shadow_steps a2a_us
  local tag="$1" shadow="$2" a2a="$3"
  echo "=== OV0B tag=$tag shadow=$shadow a2a=${a2a}us $(date +%T) ==="
  VLLM_SELF_SPEC_SHADOW_CHAIN="$shadow" W7_TAG="$tag" W7_DRAFT_QUANT=fp8 \
    W7_A2A_US="$a2a" timeout 6000 "$PY" "$SCRIPT" spec \
    > "$LOGD/${tag}_a2a${a2a}.log" 2>&1
  echo "EXIT=$? tag=$tag a2a=$a2a"
  kill_stragglers
}

kill_stragglers
run "ov0b_off_a2a0"    0 0
run "ov0b_on_a2a0"     2 0
run "ov0b_off_a2a500"  0 500
run "ov0b_on_a2a500"   2 500
echo "OV0B DONE $(date +%T)"
