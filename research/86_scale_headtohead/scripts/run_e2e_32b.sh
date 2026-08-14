#!/bin/bash
# E4: Qwen3-32B e2e arms (TP2, GPUs 0-1) -- the matched-scale head-to-head row vs KnapSpec's
# 1.28x, plus our batch/context rows. W4 draft ckpt: /data/smcho/ckpts/Qwen3-8B-W4A16-INT4
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
  local TAG="q332b_${ARM}_K${K}_b${B}_c$((CTX/1024))k"
  local LOG="$PHASE/logs/${TAG}.log"
  echo "[86e2] $TAG ($(date +%H:%M:%S))"
  kill_mine
  ( source "$P76/scripts/env_e76.sh"
    export CUDA_VISIBLE_DEVICES=0,1
    export NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo
    unset VLLM_SELF_SPEC_COMPILE_CONSISTENT VLLM_SELF_SPEC_PROFILE
    export VLLM_SELF_SPEC_DRAFT_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1
    export VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1
    export VLLM_SELF_SPEC_SHARED_KV=1
    export W7_DRAFT_QUANT= W7_DRAFT_FULL_REPLICA=0 W7_DRAFT_LOCAL_ROUTE=0 W7_DRAFT_NODE_LOCAL=0
    export W7_MODEL=Qwen/Qwen3-32B W7_TP=2 W7_EP=0 W7_TRC=0 W7_EAGER=0
    export W7_NODES=1 W7_LOCAL_WORLD=1 W7_MASTER_IP=127.0.0.1
    export W7_GPU_MEM=0.90 W7_OUTLEN=160 W7_SHORTLEN=32
    export W7_CTX_TOKENS=$CTX W7_MAX_MODEL_LEN=$((CTX + 4096)) W7_MAX_NUM_BATCHED=8192
    export W7_KS=$K W7_BATCHES=$B W7_ITERS=8 W7_WARMUP=1
    export W7_TAG="$TAG" W7_OUT="$PHASE/data/e2e"
    export W7_PROMPT_FILE="$P57/data/prompts_ondist.txt" W7_CHAT=1
    export W7_MASTER_PORT=$((19600 + K + B + RANDOM % 40))
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=180
    if [ "$MODE" = spec ]; then
      export W7_SPEC_METHOD=draft_model W7_SPEC_MODEL="/data/smcho/ckpts/Qwen3-32B-W4A16-INT4-gptq"
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
want nospec_math && run nospecm nospec 0 1 16384 W7_PROMPT_FILE=$REPO/research/79_paper/data/prompts_math.txt
want w4win_math_k4 && run w4winm spec 4 1 16384 W7_PROMPT_FILE=$REPO/research/79_paper/data/prompts_math.txt $WINENV $FIXENV
want w4win_math_k5 && run w4winm spec 5 1 16384 W7_PROMPT_FILE=$REPO/research/79_paper/data/prompts_math.txt $WINENV $FIXENV
want w4win_b1    && run w4win  spec   4 1 16384 $WINENV $FIXENV
want w4win_b1_k5 && run w4win  spec   5 1 16384 $WINENV $FIXENV
want w4win_b1_k6 && run w4win  spec   6 1 16384 $WINENV $FIXENV
want w4win_b1_t07 && run w4wint07 spec 4 1 16384 W7_TEMP=0.7 $WINENV $FIXENV
want w4win_b8    && run w4win8 spec   5 8 16384 $WINENV $FIXENV
want fp8win_b1   && run fp8win spec   5 1 16384 W7_SPEC_MODEL=Qwen/Qwen3-32B W7_DRAFT_QUANT=fp8_per_block $WINENV $FIXENV
want fp8mwin_b1_k5 && run fp8mwin spec 5 1 16384 VLLM_TEST_FORCE_FP8_MARLIN=1 W7_SPEC_MODEL=/data/smcho/ckpts/Qwen3-32B-W8A16-FP8 $WINENV $FIXENV
want fp8mwin_b1_k6 && run fp8mwin spec 6 1 16384 VLLM_TEST_FORCE_FP8_MARLIN=1 W7_SPEC_MODEL=/data/smcho/ckpts/Qwen3-32B-W8A16-FP8 $WINENV $FIXENV
want nospec_b1_t07 && run nospect07 nospec 0 1 16384 W7_TEMP=0.7
want best_b1_t07 && run bestt07 spec  5 1 16384 W7_TEMP=0.7 W7_SPEC_MODEL=Qwen/Qwen3-32B W7_DRAFT_QUANT=fp8_per_block $WINENV $FIXENV
# W4A8 queue arm (map v6.3): default disable list keeps Cutlass off -> Humming
want w4a8win_b8  && run w4a8win spec 5 8 16384 W7_SPEC_MODEL=/data/smcho/ckpts/Qwen3-32B-W4A8-gptq $WINENV $FIXENV
# b1 prose Humming arms (88 canonical sweep: Humming wins every 32B b1 cell
# on the serving driver -> re-price the KnapSpec parity cell, their 1.43)
want w4a8win_b1_k4 && run w4a8win spec 4 1 16384 W7_SPEC_MODEL=/data/smcho/ckpts/Qwen3-32B-W4A8-gptq $WINENV $FIXENV
want w4a8win_b1_k5 && run w4a8win spec 5 1 16384 W7_SPEC_MODEL=/data/smcho/ckpts/Qwen3-32B-W4A8-gptq $WINENV $FIXENV
# b16 cell (8B evidence: the W4A8 win grows with batch; K6 matches 8B b16 geometry)
want nospec_b16  && run nospec nospec 0 16 16384
want w4win_b16   && run w4win16 spec  6 16 16384 $WINENV $FIXENV
want w4a8win_b16 && run w4a8win spec 6 16 16384 W7_SPEC_MODEL=/data/smcho/ckpts/Qwen3-32B-W4A8-gptq $WINENV $FIXENV
# --- our regime bonus row
want nospec_b8   && run nospec nospec 0 8 16384
want best_b8     && run best8  spec   5 8 16384 W7_SPEC_MODEL=Qwen/Qwen3-32B W7_DRAFT_QUANT=fp8_per_block $WINENV $FIXENV
kill_mine
echo "[86e2] DONE ($(date +%H:%M:%S))"
