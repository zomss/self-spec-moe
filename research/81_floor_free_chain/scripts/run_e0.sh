#!/bin/bash
# Phase 81 E0 -- floor anatomy: kineto trace of the COMPOSED dense draft chain
# (W4-Marlin draft_model + window-KV over shared target KV), the exact config
# that measured 1.48x vs 1.91x roofline in 80-E3.
#
# Trace runs are for DECOMPOSITION ONLY (profiler tax, P74) -- never quoted
# as tok/s. Two arms: b32 K=6 (the miss cell) and b8 K=4 (the confirmed cell,
# as the contrast). GPU $E76_GPU only (6/7 are occupied by another user).
# Gate to proceed to E1: launch+orchestration >= 50% of the chain step.
set -u
PHASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$PHASE/../.." && pwd)"
P57="$REPO/research/57_large_ep_spec_strategy"
P76="$REPO/research/76_lever_latency_sweep"
PY="$REPO/.venv/bin/python"
mkdir -p "$PHASE/logs" "$PHASE/data"
ME="$(whoami)"
kill_mine(){ for p in 'trace_chain[.]py' 'EngineCor[e]' 'Worker_D[P]'; do pkill -9 -u "$ME" -f "$p" 2>/dev/null; done; sleep 5; }

trace(){  # K batch
  local K=$1 B=$2 CTX=16384
  local TD="$PHASE/data/trace_combo_K${K}_b${B}"
  mkdir -p "$TD"; rm -f "$TD"/*.json* 2>/dev/null
  local LOG="$PHASE/logs/trace_combo_K${K}_b${B}.log"
  echo "[e0] trace K=$K b=$B ($(date +%H:%M:%S))"
  kill_mine
  ( source "$P76/scripts/env_e76.sh"
    export CUDA_VISIBLE_DEVICES="$E76_GPU"
    export NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo
    unset VLLM_SELF_SPEC_COMPILE_CONSISTENT VLLM_SELF_SPEC_PROFILE
    export VLLM_SELF_SPEC_DRAFT_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1
    export VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1
    export VLLM_SELF_SPEC_SHARED_KV=1
    export VLLM_SELF_SPEC_DRAFT_KV_WINDOW=512 VLLM_SELF_SPEC_DRAFT_KV_SINKS=16
    export VLLM_DISABLED_KERNELS="MacheteLinearKernel,CutlassW4A8LinearKernel,AllSparkLinearKernel"
    export W7_MODEL=Qwen/Qwen2.5-7B-Instruct W7_SPEC_MODEL="$E76_W4" W7_EP=0
    export W7_NODES=1 W7_LOCAL_WORLD=1 W7_NODE_RANK=0 W7_MASTER_IP=127.0.0.1
    export W7_MASTER_PORT=$((17900 + K + B + RANDOM % 40))
    export W7_BATCH=$B W7_K=$K W7_CTX_TOKENS=$CTX W7_MAX_MODEL_LEN=$((CTX + 4096))
    export W7_MAX_NUM_BATCHED=8192 W7_GPU_MEM=0.90 W7_TRACE_LEN=40
    export W7_TRACE_DIR="$TD" W7_PROMPT_FILE="$P57/data/prompts_ondist.txt"
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=180
    timeout 1800 "$PY" "$PHASE/scripts/trace_chain.py"
  ) > "$LOG" 2>&1
  grep -hE 'TRACE16K|Error|Traceback' "$LOG" | tail -2 | sed 's/^/       /'
  ls -lh "$TD"/*.trace.json.gz 2>/dev/null | awk '{print "       trace:", $NF, $5}' | tail -1
  kill_mine
}

ARM="${1:-all}"
[ "$ARM" = all ] || [ "$ARM" = k6b32 ] && trace 6 32
[ "$ARM" = all ] || [ "$ARM" = k4b8 ] && trace 4 8
echo "[e0] DONE ($(date +%H:%M:%S)) -> analyze with scripts/anatomy.py"
