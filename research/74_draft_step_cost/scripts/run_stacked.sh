#!/bin/bash
# Stacked-mechanism test, batch-invariant OFF (VLLM_SELF_SPEC_COMPILE_CONSISTENT
# unset). Does window (attention slice) + fp8 (GEMM slice) COMPOSE into D<V and
# a win? Attribution arms: base (none) / window / fp8d (fp8 full-KV) / fp8dw
# (fp8+window, the stack). Clean same-session no-spec denominator per context.
# Profiler on -> draft_forward / verify per cell.
set -u
PHASE=/data/smcho/self-spec-moe/research/74_draft_step_cost
P52=/data/smcho/self-spec-moe/research/52_two_node_e2e
P57=/data/smcho/self-spec-moe/research/57_large_ep_spec_strategy
PY=/data/smcho/self-spec-moe/.venv/bin/python
ME="$(whoami)"
K=4; ITERS=2; WARMUP=1; PWARMUP=20
CONTEXTS="${CONTEXTS:-16384 32768}"
BATCHES="${BATCHES:-8}"
SPEC_ARMS="${SPEC_ARMS:-base window fp8d fp8dw}"
kill_mine(){ for p in 'w7_2node[.]py' 'EngineCor[e]' 'Worker_D[P]'; do pkill -9 -u "$ME" -f "$p" 2>/dev/null; done; sleep 5; }

arm_delta(){
  case "$1" in
    base)   echo "" ;;
    window) echo "VLLM_SELF_SPEC_DRAFT_KV_WINDOW=512 VLLM_SELF_SPEC_DRAFT_KV_SINKS=16 W7_KV_WINDOW_DEBUG=1" ;;
    fp8d)   echo "W7_DRAFT_QUANT=fp8" ;;
    fp8dw)  echo "W7_DRAFT_QUANT=fp8 VLLM_SELF_SPEC_DRAFT_KV_WINDOW=512 VLLM_SELF_SPEC_DRAFT_KV_SINKS=16 W7_KV_WINDOW_DEBUG=1" ;;
    *) echo BAD; return 1 ;;
  esac
}

run(){  # mode ctx batch arm profdir
  local MODE=$1 CTX=$2 B=$3 ARM=$4 PROFDIR=$5 MAXLEN=$(( $2 + 4096 ))
  local DELTA=""; [ "$MODE" = spec ] && DELTA="$(arm_delta "$ARM")"
  local TAG="st_${ARM}_ctx${CTX}_b${B}"; [ "$MODE" = nospec ] && TAG="st_nospec_ctx${CTX}"
  local LOG="$PHASE/logs/${TAG}.log"; local PROF=""
  [ -n "$PROFDIR" ] && { mkdir -p "$PROFDIR"; rm -f "$PROFDIR"/*.json 2>/dev/null; \
    PROF="VLLM_SELF_SPEC_PROFILE=1 VLLM_SELF_SPEC_PROFILE_OUT=$PROFDIR VLLM_SELF_SPEC_PROFILE_WARMUP=$PWARMUP"; }
  kill_mine
  ( source "$PHASE/scripts/env_cloud4.sh"
    unset VLLM_SELF_SPEC_COMPILE_CONSISTENT           # <-- batch-invariant OFF
    export W7_KS=$K W7_BATCHES="$B" W7_ITERS=$ITERS W7_WARMUP=$WARMUP W7_TAG="p74_${TAG}" \
      W7_MASTER_PORT=$((16000 + CTX/1000 + RANDOM%50)) W7_CTX_TOKENS="$CTX" W7_MAX_MODEL_LEN="$MAXLEN" \
      W7_MAX_NUM_BATCHED=8192 W7_OUT="$PHASE/data" W7_PROMPT_FILE="$P57/data/prompts_ondist.txt" W7_CHAT=1 \
      $DELTA $PROF
    W7_NODE_RANK=0 timeout 1200 "$PY" "$P52/scripts/w7_2node.py" "$MODE" ) > "$LOG" 2>&1
  echo "[st] $TAG:"; grep -hE "W7-2N.*batch=|GPU KV cache size" "$LOG" | tail -4
  kill_mine
}

for CTX in $CONTEXTS; do
  run nospec "$CTX" "$(echo $BATCHES | tr ' ' ',')" nospec ""
  for ARM in $SPEC_ARMS; do
    for B in $BATCHES; do
      run spec "$CTX" "$B" "$ARM" "$PHASE/data/prof_st_${ARM}_ctx${CTX}_b${B}"
    done
  done
done
echo "[st] COMPLETE ($(date +%H:%M:%S))"
