#!/bin/bash
# Extend the win to 16k via K-tuning + window. window draft @ K=2,3 at 16k/32k
# b8, clean (no profiler, batch-invariant off), ITERS=4 for a stable slope.
# base @ K=2 @ 32k for comparison. no-spec refs: 599 (16k) / 436 (32k).
set -u
PHASE=/data/smcho/self-spec-moe/research/74_draft_step_cost
P52=/data/smcho/self-spec-moe/research/52_two_node_e2e
P57=/data/smcho/self-spec-moe/research/57_large_ep_spec_strategy
PY=/data/smcho/self-spec-moe/.venv/bin/python
ME="$(whoami)"
kill_mine(){ for p in 'w7_2node[.]py' 'EngineCor[e]' 'Worker_D[P]'; do pkill -9 -u "$ME" -f "$p" 2>/dev/null; done; sleep 5; }
run(){  # arm K ctx
  local ARM=$1 K=$2 CTX=$3 MAXLEN=$(( $3 + 4096 ))
  local DELTA=""; [ "$ARM" = window ] && DELTA="VLLM_SELF_SPEC_DRAFT_KV_WINDOW=512 VLLM_SELF_SPEC_DRAFT_KV_SINKS=16"
  local LOG="$PHASE/logs/wk_${ARM}_K${K}_ctx${CTX}_b8.log"
  kill_mine
  ( source "$PHASE/scripts/env_cloud4.sh"
    unset VLLM_SELF_SPEC_COMPILE_CONSISTENT
    export W7_KS=$K W7_BATCHES=8 W7_ITERS=4 W7_WARMUP=1 W7_TAG="p74_wk_${ARM}_K${K}_ctx${CTX}_b8" \
      W7_MASTER_PORT=$((17400 + K + CTX/1000 + RANDOM%25)) W7_CTX_TOKENS="$CTX" W7_MAX_MODEL_LEN="$MAXLEN" \
      W7_MAX_NUM_BATCHED=8192 W7_OUT="$PHASE/data" W7_PROMPT_FILE="$P57/data/prompts_ondist.txt" W7_CHAT=1 $DELTA
    W7_NODE_RANK=0 timeout 1200 "$PY" "$P52/scripts/w7_2node.py" spec ) > "$LOG" 2>&1
  echo "[wk] ${ARM} K=${K} ctx${CTX}: $(grep -hE 'W7-2N.*batch=' "$LOG" | tail -1)"
  kill_mine
}
run window 2 16384
run window 3 16384
run base   2 16384
run window 2 32768
run window 3 32768
echo "[wk] DONE ($(date +%H:%M:%S))"
