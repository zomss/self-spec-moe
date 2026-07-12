#!/bin/bash
# Phase 80 E3 -- e2e check of map v4's composed headline (dense q_int4+win512):
#   b32/16k: registered 1.91x @ gamma*=6 (measbeta x measR -- the map's
#            strongest-provenance claim; expect ~86% delivery at g6 -> ~1.64x,
#            which would still beat the SINGLE-lever measured 1.54x)
#   b8/16k:  registered 1.55x @ gamma*=4 (dense delivery ~100% at g4 in 76-E3)
#
# Draft = W4 checkpoint via draft_model (P75-E3 method, Marlin forced) +
# window-KV over the shared target KV (P74/76-E3 knobs). Real spec-vs-nospec
# tok/s in the W7 harness. GPUs: $E76_GPU only.
# Run BY PATH: bash scripts/e3_combo.sh
set -u
PHASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$PHASE/../.." && pwd)"
P52="$REPO/research/52_two_node_e2e"
P57="$REPO/research/57_large_ep_spec_strategy"
P76="$REPO/research/76_lever_latency_sweep"
PY="$REPO/.venv/bin/python"
mkdir -p "$PHASE/logs" "$PHASE/data/e3"
ME="$(whoami)"
kill_mine(){ for p in 'w7_2node[.]py' 'EngineCor[e]' 'Worker_D[P]' 'vllm[.]entrypoints'; do pkill -9 -u "$ME" -f "$p" 2>/dev/null; done; sleep 5; }

run(){  # mode K batch
  local MODE=$1 K=$2 B=$3 CTX=16384
  local TAG="e3c_dense_${MODE}_K${K}_b${B}_c16k"
  local LOG="$PHASE/logs/${TAG}.log"
  echo "[e3c] $TAG ($(date +%H:%M:%S))"
  kill_mine
  ( source "$P76/scripts/env_e76.sh"
    export CUDA_VISIBLE_DEVICES="$E76_GPU"
    export NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo
    unset VLLM_SELF_SPEC_COMPILE_CONSISTENT VLLM_SELF_SPEC_PROFILE
    export VLLM_SELF_SPEC_DRAFT_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1
    export VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1
    export VLLM_SELF_SPEC_SHARED_KV=1
    export W7_DRAFT_QUANT= W7_DRAFT_FULL_REPLICA=0 W7_DRAFT_LOCAL_ROUTE=0 W7_DRAFT_NODE_LOCAL=0
    export W7_MODEL=Qwen/Qwen2.5-7B-Instruct W7_TP=1 W7_EP=0 W7_TRC=0 W7_EAGER=0
    export W7_NODES=1 W7_LOCAL_WORLD=1 W7_MASTER_IP=127.0.0.1
    export W7_GPU_MEM=0.90 W7_OUTLEN=160 W7_SHORTLEN=32
    export W7_CTX_TOKENS=$CTX W7_MAX_MODEL_LEN=$((CTX + 4096)) W7_MAX_NUM_BATCHED=8192
    export W7_KS=$K W7_BATCHES=$B W7_ITERS=4 W7_WARMUP=1
    export W7_TAG="$TAG" W7_OUT="$PHASE/data/e3"
    export W7_PROMPT_FILE="$P57/data/prompts_ondist.txt" W7_CHAT=1
    export W7_MASTER_PORT=$((17800 + K + B + RANDOM % 40))
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=180
    if [ "$MODE" = spec ]; then
      # the COMPOSED draft: W4 ckpt (Marlin forced) + window-KV over target KV
      export W7_SPEC_METHOD=draft_model
      export W7_SPEC_MODEL="$E76_W4"
      export VLLM_DISABLED_KERNELS="MacheteLinearKernel,CutlassW4A8LinearKernel,AllSparkLinearKernel"
      export VLLM_SELF_SPEC_DRAFT_KV_WINDOW=512 VLLM_SELF_SPEC_DRAFT_KV_SINKS=16
    fi
    W7_NODE_RANK=0 timeout 2400 "$PY" "$P52/scripts/w7_2node.py" "$MODE"
  ) > "$LOG" 2>&1
  grep -hE 'W7-2N.*batch=' "$LOG" | tail -2 | sed 's/^/       /'
  kill_mine
}

echo "[e3c] === dense q_int4+win512: map v4 1.91x @b32/16k (g6), 1.55x @b8/16k (g4) ==="
run nospec 0 32
run spec 6 32
run spec 4 32
run nospec 0 8
run spec 4 8
kill_mine
echo "[e3c] DONE ($(date +%H:%M:%S))"
