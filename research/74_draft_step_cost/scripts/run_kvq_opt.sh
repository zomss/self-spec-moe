#!/bin/bash
# Optimize/confirm the KV-QUANT draft strategy. Agent-verified: fp8_e4m3 is the
# ONLY fp8 KV that stays on the FA3 fast path (e5m2/per_token_head/turboquant fall
# off FA3). KV-quant is GLOBAL under SHARED_KV (quantizes verify's pool too) ->
# bounded-lossy, and it speeds the no-spec baseline too, so compare spec-vs-nospec
# BOTH under fp8 KV. 16k b8, K=2, NO window, clean. Refs (bf16 KV): nospec=599, base=592.
set -u
PHASE=/data/smcho/self-spec-moe/research/74_draft_step_cost
P52=/data/smcho/self-spec-moe/research/52_two_node_e2e
P57=/data/smcho/self-spec-moe/research/57_large_ep_spec_strategy
PY=/data/smcho/self-spec-moe/.venv/bin/python
ME="$(whoami)"
kill_mine(){ for p in 'w7_2node[.]py' 'EngineCor[e]' 'Worker_D[P]'; do pkill -9 -u "$ME" -f "$p" 2>/dev/null; done; sleep 5; }
run(){  # mode
  local MODE=$1
  local LOG="$PHASE/logs/kvq_${MODE}_fp8e4m3_K2_16k_b8.log"
  kill_mine
  ( source "$PHASE/scripts/env_cloud4.sh"
    unset VLLM_SELF_SPEC_COMPILE_CONSISTENT
    export W7_KV_CACHE_DTYPE=fp8_e4m3
    export W7_KS=2 W7_BATCHES=8 W7_ITERS=4 W7_WARMUP=1 W7_TAG="p74_kvq_${MODE}_fp8e4m3_K2_16k_b8" \
      W7_MASTER_PORT=$((17800 + RANDOM%40)) W7_CTX_TOKENS=16384 W7_MAX_MODEL_LEN=20480 \
      W7_MAX_NUM_BATCHED=8192 W7_OUT="$PHASE/data" W7_PROMPT_FILE="$P57/data/prompts_ondist.txt" W7_CHAT=1
    W7_NODE_RANK=0 timeout 1200 "$PY" "$P52/scripts/w7_2node.py" "$MODE" ) > "$LOG" 2>&1
  echo "[kvq] ${MODE}: $(grep -hE 'W7-2N.*batch=' "$LOG" | tail -1)"
  # confirm which attention backend engaged (must be FLASH_ATTN / FA3, NOT FlashInfer)
  grep -hiE 'Using .*attention backend|FlashAttn|FLASH_ATTN|FlashInfer|calculate.kv.scales|kv.cache.*dtype|fp8' "$LOG" | grep -iE 'backend|fp8|scale' | tail -3
  kill_mine
}
run nospec
run spec
echo "[kvq] DONE ($(date +%H:%M:%S))"
