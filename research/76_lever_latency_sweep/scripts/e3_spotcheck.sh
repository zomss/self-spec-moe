#!/bin/bash
# Phase 76 E3 -- e2e spot-check of the two UNMEASURED strategy-map cells:
#   1. dense Qwen2.5-7B  b32/16k window  -> map predicts 1.40x @ gamma*=4
#   2. MoE  Qwen3-30B    b32/32k window  -> map predicts 1.90x @ gamma*=6
# Real spec-vs-nospec tok/s via the W7 harness (P74/P75's e2e method).
# Each cell also runs a K=3 arm: the map's gamma* is a roofline; the PIECEWISE
# harness taxes deep chains (P74: measured K* < formula gamma* is possible).
# Expect ~90% of map values on this harness (P75 E3 delivery).
#
# GPUs: dense on $E76_GPU, MoE DP4/EP4 on $E76_MOE_GPUS (0,1,6,7 -- NOT the
# P74 env_cloud4 default 4,5,6,7; GPUs 2-5 are reserved).
# Run BY PATH: bash scripts/e3_spotcheck.sh [dense|moe|all]
set -u
PHASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$PHASE/../.." && pwd)"
P52="$REPO/research/52_two_node_e2e"
P57="$REPO/research/57_large_ep_spec_strategy"
P74="$REPO/research/74_draft_step_cost"
PY="$REPO/.venv/bin/python"
GROUP="${1:-all}"
mkdir -p "$PHASE/logs" "$PHASE/data/e3"
ME="$(whoami)"
kill_mine(){ for p in 'w7_2node[.]py' 'EngineCor[e]' 'Worker_D[P]' 'vllm[.]entrypoints'; do pkill -9 -u "$ME" -f "$p" 2>/dev/null; done; sleep 5; }

WIN="VLLM_SELF_SPEC_DRAFT_KV_WINDOW=512 VLLM_SELF_SPEC_DRAFT_KV_SINKS=16"

dense_run(){  # mode K
  local MODE=$1 K=$2 CTX=16384
  local TAG="e3_dense_${MODE}_K${K}_b32_c16k"
  local LOG="$PHASE/logs/${TAG}.log"
  echo "[e3] $TAG ($(date +%H:%M:%S))"
  kill_mine
  ( source "$PHASE/scripts/env_e76.sh"     # /data caches + GPU constraint
    export CUDA_VISIBLE_DEVICES="$E76_GPU"
    export NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo
    unset VLLM_SELF_SPEC_COMPILE_CONSISTENT VLLM_SELF_SPEC_PROFILE
    # P74/47/52 self-spec stack + SHARED_KV (window reads the target's KV)
    export VLLM_SELF_SPEC_DRAFT_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1
    export VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1
    export VLLM_SELF_SPEC_SHARED_KV=1
    export W7_DRAFT_QUANT= W7_DRAFT_FULL_REPLICA=0 W7_DRAFT_LOCAL_ROUTE=0 W7_DRAFT_NODE_LOCAL=0
    export W7_MODEL=Qwen/Qwen2.5-7B-Instruct W7_TP=1 W7_EP=0 W7_TRC=0 W7_EAGER=0
    export W7_NODES=1 W7_LOCAL_WORLD=1 W7_MASTER_IP=127.0.0.1
    export W7_GPU_MEM=0.90 W7_OUTLEN=160 W7_SHORTLEN=32
    export W7_CTX_TOKENS=$CTX W7_MAX_MODEL_LEN=$((CTX + 4096)) W7_MAX_NUM_BATCHED=8192
    export W7_KS=$K W7_BATCHES=32 W7_ITERS=4 W7_WARMUP=1
    export W7_TAG="$TAG" W7_OUT="$PHASE/data/e3"
    export W7_PROMPT_FILE="$P57/data/prompts_ondist.txt" W7_CHAT=1
    export W7_MASTER_PORT=$((17600 + K + RANDOM % 50))
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=180
    [ "$MODE" = spec ] && for kv in $WIN; do export "${kv?}"; done
    W7_NODE_RANK=0 timeout 2400 "$PY" "$P52/scripts/w7_2node.py" "$MODE"
  ) > "$LOG" 2>&1
  grep -hE 'W7-2N.*batch=' "$LOG" | tail -2 | sed 's/^/       /'
  kill_mine
}

moe_run(){  # mode K
  local MODE=$1 K=$2 CTX=32768
  local TAG="e3_moe_${MODE}_K${K}_b32_c32k"
  local LOG="$PHASE/logs/${TAG}.log"
  echo "[e3] $TAG ($(date +%H:%M:%S))"
  kill_mine
  ( source "$P74/scripts/env_cloud4.sh"    # proven MoE DP4/EP4 self-spec stack
    source "$PHASE/scripts/env_e76.sh"     # overrides: /data caches, nvcc, GPUs
    export CUDA_VISIBLE_DEVICES="$E76_MOE_GPUS"          # NOT 4,5,6,7
    unset VLLM_SELF_SPEC_COMPILE_CONSISTENT VLLM_SELF_SPEC_PROFILE  # P74 hygiene
    export W7_CTX_TOKENS=$CTX W7_MAX_MODEL_LEN=$((CTX + 4096)) W7_MAX_NUM_BATCHED=8192
    export W7_KS=$K W7_BATCHES=32 W7_ITERS=4 W7_WARMUP=1
    export W7_TAG="$TAG" W7_OUT="$PHASE/data/e3"
    export W7_PROMPT_FILE="$P57/data/prompts_ondist.txt" W7_CHAT=1
    export W7_MASTER_PORT=$((17700 + K + RANDOM % 50))
    [ "$MODE" = spec ] && for kv in $WIN; do export "${kv?}"; done
    W7_NODE_RANK=0 timeout 3600 "$PY" "$P52/scripts/w7_2node.py" "$MODE"
  ) > "$LOG" 2>&1
  grep -hE 'W7-2N.*batch=' "$LOG" | tail -2 | sed 's/^/       /'
  kill_mine
}

if [ "$GROUP" = all ] || [ "$GROUP" = dense ]; then
  echo "[e3] === DENSE b32/16k window (map: 1.40x @ g4) ==="
  dense_run nospec 0
  dense_run spec 4
  dense_run spec 3
fi
if [ "$GROUP" = all ] || [ "$GROUP" = moe ]; then
  echo "[e3] === MoE b32/32k window (map: 1.90x @ g6) ==="
  moe_run nospec 0
  moe_run spec 6
  moe_run spec 3
fi
kill_mine
echo "[e3] DONE ($(date +%H:%M:%S)) -- logs/e3_*, data/e3/"
