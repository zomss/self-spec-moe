#!/bin/bash
# Torch-profiler trace of the DRAFT forward for each method at 16k b8,
# batch-invariant OFF (cuBLAS). Arms: base / window / fp8d. Diff the per-
# category GPU time to explain why each method's slice reduction is limited.
set -u
PHASE=/data/smcho/self-spec-moe/research/74_draft_step_cost
TR=/data/smcho/self-spec-moe/research/64_window_e2e/scripts/w7_trace16k.py
P57=/data/smcho/self-spec-moe/research/57_large_ep_spec_strategy
PY=/data/smcho/self-spec-moe/.venv/bin/python
ME="$(whoami)"
kill_mine(){ for p in 'w7_trace16k[.]py' 'w7_2node[.]py' 'EngineCor[e]' 'Worker_D[P]'; do pkill -9 -u "$ME" -f "$p" 2>/dev/null; done; sleep 5; }

for ARM in base window fp8d; do
  DELTA=""
  case "$ARM" in
    window) DELTA="VLLM_SELF_SPEC_DRAFT_KV_WINDOW=512 VLLM_SELF_SPEC_DRAFT_KV_SINKS=16" ;;
    fp8d)   DELTA="W7_DRAFT_QUANT=fp8" ;;
  esac
  TRACE_DIR="$PHASE/data/trace_m_${ARM}_16k_b8"; mkdir -p "$TRACE_DIR"; rm -f "$TRACE_DIR"/*.json* 2>/dev/null
  LOG="$PHASE/logs/trace_m_${ARM}_16k_b8.log"
  kill_mine
  ( source "$PHASE/scripts/env_cloud4.sh"
    unset VLLM_SELF_SPEC_COMPILE_CONSISTENT     # cuBLAS, no batch-invariant
    export W7_NODES=1 W7_LOCAL_WORLD=4 W7_BATCH=8 W7_K=4 W7_CTX_TOKENS=16384 \
      W7_MAX_MODEL_LEN=20480 W7_MAX_NUM_BATCHED=8192 W7_TRACE_LEN=40 \
      W7_MASTER_PORT=$((16600 + RANDOM%50)) W7_TRACE_DIR="$TRACE_DIR" \
      W7_PROMPT_FILE="$P57/data/prompts_ondist.txt" $DELTA
    W7_NODE_RANK=0 timeout 900 "$PY" "$TR" ) > "$LOG" 2>&1
  echo "[trm] $ARM: $(grep -hE 'TRACE16K|Draft model quantization|kv-window' "$LOG" | tail -2)"
  ls -lh "$TRACE_DIR"/*.pt.trace.json.gz 2>/dev/null | tail -1
  kill_mine
done
echo "[trm] DONE ($(date +%H:%M:%S))"
