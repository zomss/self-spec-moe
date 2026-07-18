#!/bin/bash
# E2: Qwen3-8B e2e arms -- the matched-scale head-to-head row vs KnapSpec's
# 1.28x, plus our batch/context rows. W4 draft ckpt: ~/ckpts/Qwen3-8B-W4A16-INT4
# (P74, data-free RTN -- matches the measured beta 0.9523 fake-quant semantics).
# Run BY PATH: bash scripts/run_e2e_8b.sh [filter]
set -u
PHASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$PHASE/../.." && pwd)"
P52="$REPO/research/52_two_node_e2e"
P57="$REPO/research/57_large_ep_spec_strategy"
P76="$REPO/research/76_lever_latency_sweep"
PY="$REPO/.venv/bin/python"
mkdir -p "$PHASE/logs" "$PHASE/data/e2e"
ME="$(whoami)"
kill_mine(){ for p in 'w7_2node[.]py' 'EngineCor[e]' 'Worker_D[P]'; do pkill -9 -u "$ME" -f "$p" 2>/dev/null; done; sleep 5; }

run(){  # arm mode K B ctx [extra-env...]
  local ARM=$1 MODE=$2 K=$3 B=$4 CTX=$5; shift 5
  local TAG="q38b_${ARM}_K${K}_b${B}_c$((CTX/1024))k"
  local LOG="$PHASE/logs/${TAG}.log"
  echo "[86e2] $TAG ($(date +%H:%M:%S))"
  kill_mine
  ( source "$P76/scripts/env_e76.sh"
    export CUDA_VISIBLE_DEVICES=6
    export NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo
    unset VLLM_SELF_SPEC_COMPILE_CONSISTENT VLLM_SELF_SPEC_PROFILE
    export VLLM_SELF_SPEC_DRAFT_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1
    export VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1
    export VLLM_SELF_SPEC_SHARED_KV=1
    export W7_DRAFT_QUANT= W7_DRAFT_FULL_REPLICA=0 W7_DRAFT_LOCAL_ROUTE=0 W7_DRAFT_NODE_LOCAL=0
    export W7_MODEL=Qwen/Qwen3-8B W7_TP=1 W7_EP=0 W7_TRC=0 W7_EAGER=0
    export W7_NODES=1 W7_LOCAL_WORLD=1 W7_MASTER_IP=127.0.0.1
    export W7_GPU_MEM=0.90 W7_OUTLEN=160 W7_SHORTLEN=32
    export W7_CTX_TOKENS=$CTX W7_MAX_MODEL_LEN=$((CTX + 4096)) W7_MAX_NUM_BATCHED=8192
    export W7_KS=$K W7_BATCHES=$B W7_ITERS=8 W7_WARMUP=1
    export W7_TAG="$TAG" W7_OUT="$PHASE/data/e2e"
    export W7_PROMPT_FILE="$P57/data/prompts_ondist.txt" W7_CHAT=1
    export W7_MASTER_PORT=$((19600 + K + B + RANDOM % 40))
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=180
    if [ "$MODE" = spec ]; then
      export W7_SPEC_METHOD=draft_model W7_SPEC_MODEL="$HOME/ckpts/Qwen3-8B-W4A16-INT4"
      export VLLM_DISABLED_KERNELS="MacheteLinearKernel,CutlassW4A8LinearKernel,AllSparkLinearKernel"
    fi
    for kv in "$@"; do export "${kv?}"; done
    W7_NODE_RANK=0 timeout 2400 "$PY" "$P52/scripts/w7_2node.py" "$MODE"
  ) > "$LOG" 2>&1
  grep -hE 'W7-2N.*batch=' "$LOG" | tail -1 | sed "s/^/       [$TAG] /"
  kill_mine
}

WINENV="VLLM_SELF_SPEC_DRAFT_KV_WINDOW=512 VLLM_SELF_SPEC_DRAFT_KV_SINKS=16"
FIXENV="VLLM_SELF_SPEC_DRAFT_FULLCG=1 VLLM_SELF_SPEC_DRAFT_STEP0_FULL_CG=1 VLLM_SELF_SPEC_CPU_ORCH=1 W7_ASYNC_SCHED=1 VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE=1"
FILTER="${1:-.}"
want(){ echo "$1" | grep -qE "$FILTER"; }
# --- their setting: b1, greedy and T=0.7
want nospec_b1   && run nospec nospec 0 1 16384
want w4_b1       && run w4     spec   4 1 16384
want w4_b1_t07   && run w4t07  spec   4 1 16384 W7_TEMP=0.7
want nospec_b1_t07 && run nospect07 nospec 0 1 16384 W7_TEMP=0.7
want w4win_b1    && run w4win  spec   4 1 16384 $WINENV $FIXENV
want w4win_b1_t07 && run w4wint07 spec 4 1 16384 W7_TEMP=0.7 $WINENV $FIXENV
want w4_b1_2k    && run w4s    spec   4 1 2048 
want w4win_b1_2k && run w4wins spec   4 1 2048 $WINENV $FIXENV
want nospec_b1_2k && run nospecs nospec 0 1 2048
# --- our regimes (no KnapSpec counterpart)
# W4A8 arms: re-enable CutlassW4A8 (the default spec-env disable exists to
# force Marlin on W4A16 arms; Humming-fallback b16 preserved as *_humming)
W4A8ENV="W7_SPEC_MODEL=$HOME/ckpts/Qwen3-8B-W4A8-gptq VLLM_DISABLED_KERNELS=MacheteLinearKernel,AllSparkLinearKernel"
want w4a8win_b8  && run w4a8win spec 4 8 16384 $W4A8ENV $WINENV $FIXENV
want w4a8win_b16 && run w4a8win spec 6 16 16384 $W4A8ENV $WINENV $FIXENV
# Humming realization (default disable list keeps Cutlass off)
want w4a8hwin_b8  && run w4a8hwin spec 4 8 16384 W7_SPEC_MODEL=$HOME/ckpts/Qwen3-8B-W4A8-gptq $WINENV $FIXENV
want w4a8hwin_b16 && run w4a8hwin spec 6 16 16384 W7_SPEC_MODEL=$HOME/ckpts/Qwen3-8B-W4A8-gptq $WINENV $FIXENV
want nospec_b8   && run nospec nospec 0 8 16384
want w4win_b8    && run w4win  spec   4 8 16384 $WINENV $FIXENV
want nospec_b32  && run nospec nospec 0 32 16384
want w4win_b32   && run w4win  spec   6 32 16384 $WINENV $FIXENV
want nospec_b16  && run nospec nospec 0 16 16384
want w4win_b16   && run w4win  spec   6 16 16384 $WINENV $FIXENV
kill_mine
echo "[86e2] DONE ($(date +%H:%M:%S))"
