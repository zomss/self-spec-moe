#!/bin/bash
# Phase 79 (paper, sec4-F3): the composed over-capacity cell -- dense b32/32k.
# bf16 nospec thrashes here (76-E0: 7/32 queued, TPOT 117-140ms); the composed
# W4+window draft on the FIXED chain should hold the batch AND win.
# Derived from 81/scripts/run_e3.sh with CTX=32768:
#   paged-FA3 scratchpad chain (E1b) + compacted q=1 step-0 (E2b fix)
#   + CPU_ORCH fast parse + async scheduling.
# 80-E3 reference (broken chain): b32/16k K6 1.48x, K4 ~; b8/16k K4 1.55x.
# E2b spot: b32/16k K6 = 4148 tok/s = 1.88x.
# Cells run IN PARALLEL, one per GPU (0-3; user-cleared 2026-07-13).
# Run BY PATH: bash scripts/run_e3.sh
set -u
PHASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$PHASE/../.." && pwd)"
P52="$REPO/research/52_two_node_e2e"
P57="$REPO/research/57_large_ep_spec_strategy"
P76="$REPO/research/76_lever_latency_sweep"
PY="$REPO/.venv/bin/python"
mkdir -p "$PHASE/logs" "$PHASE/data/e3_32k"
ME="$(whoami)"
kill_mine(){ for p in 'w7_2node[.]py' 'EngineCor[e]' 'Worker_D[P]' 'vllm[.]entrypoints'; do pkill -9 -u "$ME" -f "$p" 2>/dev/null; done; sleep 5; }

run(){  # gpu arm mode K batch [extra-env...]
  local GPU=$1 ARM=$2 MODE=$3 K=$4 B=$5; shift 5
  local CTX=32000    # qwen2.5 max_position_embeddings=32768; leave outlen room
  local TAG="e3x32_dense_${ARM}_K${K}_b${B}_c32k"
  local LOG="$PHASE/logs/${TAG}.log"
  echo "[e3] $TAG (GPU $GPU, $(date +%H:%M:%S))"
  ( source "$P76/scripts/env_e76.sh"
    export CUDA_VISIBLE_DEVICES="$GPU"
    export NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo
    unset VLLM_SELF_SPEC_COMPILE_CONSISTENT VLLM_SELF_SPEC_PROFILE
    export VLLM_SELF_SPEC_DRAFT_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1
    export VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1
    export VLLM_SELF_SPEC_SHARED_KV=1
    export W7_DRAFT_QUANT= W7_DRAFT_FULL_REPLICA=0 W7_DRAFT_LOCAL_ROUTE=0 W7_DRAFT_NODE_LOCAL=0
    export W7_MODEL=Qwen/Qwen2.5-7B-Instruct W7_TP=1 W7_EP=0 W7_TRC=0 W7_EAGER=0
    export W7_NODES=1 W7_LOCAL_WORLD=1 W7_MASTER_IP=127.0.0.1
    export W7_GPU_MEM=0.90 W7_OUTLEN=160 W7_SHORTLEN=32
    export W7_CTX_TOKENS=$CTX W7_MAX_MODEL_LEN=32768 W7_MAX_NUM_BATCHED=8192
    export W7_KS=$K W7_BATCHES=$B W7_ITERS=4 W7_WARMUP=1
    export W7_TAG="$TAG" W7_OUT="$PHASE/data/e3_32k"
    export W7_PROMPT_FILE="$P57/data/prompts_ondist.txt" W7_CHAT=1
    export W7_MASTER_PORT=$((18900 + GPU * 37 + K + B))
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=180
    if [ "$MODE" = spec ]; then
      # composed draft (W4-Marlin + window-KV) on the FIXED chain
      export W7_SPEC_METHOD=draft_model W7_SPEC_MODEL="$E76_W4"
      export VLLM_DISABLED_KERNELS="MacheteLinearKernel,CutlassW4A8LinearKernel,AllSparkLinearKernel"
      export VLLM_SELF_SPEC_DRAFT_KV_WINDOW=512 VLLM_SELF_SPEC_DRAFT_KV_SINKS=16
      export VLLM_SELF_SPEC_DRAFT_FULLCG=1 VLLM_SELF_SPEC_DRAFT_STEP0_FULL_CG=1
      export VLLM_SELF_SPEC_CPU_ORCH=1 W7_ASYNC_SCHED=1
      export VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE=1
    fi
    for kv in "$@"; do export "${kv?}"; done
    W7_NODE_RANK=0 timeout 2400 "$PY" "$P52/scripts/w7_2node.py" "$MODE"
  ) > "$LOG" 2>&1
  grep -hE 'W7-2N.*batch=' "$LOG" | tail -1 | sed "s/^/       [$TAG] /"
}

kill_mine
FILTER="${1:-b32}"
if [ "$FILTER" = b32 ]; then
echo "[e3x32] === b32/32k: nospec (over-capacity doc) + composed spec ==="
run 0 nospec nospec 0 32 &
run 1 spec   spec   6 32 &
run 2 spec   spec   4 32 &
wait
fi
if [ "$FILTER" = b16 ]; then
echo "[e3x32] === b16/32k: the clean long-ctx composed cell (fits KV pool) ==="
run 0 nospec nospec 0 16 &
run 1 spec   spec   6 16 &
run 2 spec   spec   4 16 &
wait
fi
kill_mine
echo "[e3] DONE ($(date +%H:%M:%S))"
