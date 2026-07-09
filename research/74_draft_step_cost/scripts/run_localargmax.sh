#!/bin/bash
# Test the cheap base-plumbing win: use_local_argmax_reduction (fused
# vocab-parallel argmax, no full-logits materialization between forwards).
# base + window at 32k b8, COMPILE_CONSISTENT off, W7_LOCAL_ARGMAX on.
# Compare accept (parity check) + tok/s to: base 311/acc5.00, window 386/acc4.65.
set -u
PHASE=/data/smcho/self-spec-moe/research/74_draft_step_cost
P52=/data/smcho/self-spec-moe/research/52_two_node_e2e
P57=/data/smcho/self-spec-moe/research/57_large_ep_spec_strategy
PY=/data/smcho/self-spec-moe/.venv/bin/python
ME="$(whoami)"
kill_mine(){ for p in 'w7_2node[.]py' 'EngineCor[e]' 'Worker_D[P]'; do pkill -9 -u "$ME" -f "$p" 2>/dev/null; done; sleep 5; }
for ARM in base window; do
  DELTA=""; [ "$ARM" = window ] && DELTA="VLLM_SELF_SPEC_DRAFT_KV_WINDOW=512 VLLM_SELF_SPEC_DRAFT_KV_SINKS=16"
  PROFDIR="$PHASE/data/prof_la_${ARM}_ctx32768_b8"; mkdir -p "$PROFDIR"; rm -f "$PROFDIR"/*.json 2>/dev/null
  LOG="$PHASE/logs/la_${ARM}_ctx32768_b8.log"
  kill_mine
  ( source "$PHASE/scripts/env_cloud4.sh"
    unset VLLM_SELF_SPEC_COMPILE_CONSISTENT
    export W7_LOCAL_ARGMAX=1
    export W7_KS=4 W7_BATCHES=8 W7_ITERS=2 W7_WARMUP=1 W7_TAG="p74_la_${ARM}_ctx32768_b8" \
      W7_MASTER_PORT=$((16400 + RANDOM%50)) W7_CTX_TOKENS=32768 W7_MAX_MODEL_LEN=36864 \
      W7_MAX_NUM_BATCHED=8192 W7_OUT="$PHASE/data" W7_PROMPT_FILE="$P57/data/prompts_ondist.txt" W7_CHAT=1 \
      $DELTA VLLM_SELF_SPEC_PROFILE=1 VLLM_SELF_SPEC_PROFILE_OUT="$PROFDIR" VLLM_SELF_SPEC_PROFILE_WARMUP=20
    W7_NODE_RANK=0 timeout 1200 "$PY" "$P52/scripts/w7_2node.py" spec ) > "$LOG" 2>&1
  echo "[la] $ARM 32k b8 (local_argmax on):"; grep -hE "W7-2N.*batch=|local.argmax|use_local" "$LOG" | tail -3
  kill_mine
done
echo "[la] DONE ($(date +%H:%M:%S))"
