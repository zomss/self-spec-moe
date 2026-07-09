#!/bin/bash
# Test: spec WITHOUT VLLM_SELF_SPEC_COMPILE_CONSISTENT (batch-invariant OFF) so
# the draft+verify use fast cuBLAS instead of the slow Triton matmul_persistent.
# Measures draft_forward + accept + tok/s at 2k b8 and 16k b8.
set -u
PHASE=/data/smcho/self-spec-moe/research/74_draft_step_cost
P52=/data/smcho/self-spec-moe/research/52_two_node_e2e
P57=/data/smcho/self-spec-moe/research/57_large_ep_spec_strategy
PY=/data/smcho/self-spec-moe/.venv/bin/python
ME="$(whoami)"
kill_mine(){ for p in 'w7_2node[.]py' 'EngineCor[e]' 'Worker_D[P]'; do pkill -9 -u "$ME" -f "$p" 2>/dev/null; done; sleep 5; }

for pt in 2048:8 16384:8; do
  CTX="${pt%%:*}"; B="${pt##*:}"; MAXLEN=$((CTX+4096))
  PROFDIR="$PHASE/data/prof_noconsist_ctx${CTX}_b${B}"
  mkdir -p "$PROFDIR"; rm -f "$PROFDIR"/*.json 2>/dev/null
  LOG="$PHASE/logs/noconsist_ctx${CTX}_b${B}.log"
  kill_mine
  ( source "$PHASE/scripts/env_cloud4.sh"
    unset VLLM_SELF_SPEC_COMPILE_CONSISTENT
    export W7_KS=4 W7_BATCHES="$B" W7_ITERS=2 W7_WARMUP=1 \
      W7_TAG="p74_noconsist_ctx${CTX}_b${B}" W7_MASTER_PORT=$((15800 + CTX/1000)) \
      W7_CTX_TOKENS="$CTX" W7_MAX_MODEL_LEN="$MAXLEN" W7_MAX_NUM_BATCHED=8192 \
      W7_OUT="$PHASE/data" W7_PROMPT_FILE="$P57/data/prompts_ondist.txt" W7_CHAT=1 \
      VLLM_SELF_SPEC_PROFILE=1 VLLM_SELF_SPEC_PROFILE_OUT="$PROFDIR" VLLM_SELF_SPEC_PROFILE_WARMUP=20
    W7_NODE_RANK=0 timeout 1200 "$PY" "$P52/scripts/w7_2node.py" spec ) > "$LOG" 2>&1
  echo "[noconsist] ctx${CTX} b${B}:"
  grep -hE "W7-2N.*batch=|batch.invariant|BATCH_INVARIANT" "$LOG" | tail -3
  kill_mine
done
echo "[noconsist] DONE ($(date +%H:%M:%S))"
