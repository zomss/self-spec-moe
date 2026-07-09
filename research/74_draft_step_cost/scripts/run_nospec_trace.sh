#!/bin/bash
set -u
PHASE=/data/smcho/self-spec-moe/research/74_draft_step_cost
PY=/data/smcho/self-spec-moe/.venv/bin/python
TRACE_DIR="$PHASE/data/trace_nospec_2k_b8"
LOG="$PHASE/logs/trace_nospec_2k_b8.log"
mkdir -p "$TRACE_DIR"; rm -f "$TRACE_DIR"/*.json* 2>/dev/null
for p in $(pgrep -u "$(whoami)" -f "Worker_DP|EngineCore" 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
sleep 5
source "$PHASE/scripts/env_cloud4.sh"
# nospec: drop the self-spec stack (inert without a spec config; keeps it clean)
unset VLLM_SELF_SPEC_DRAFT_FULL_CG VLLM_SELF_SPEC_COMPILE_CONSISTENT \
  VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU \
  VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD VLLM_SELF_SPEC_SHARED_KV
export W7_NOSPEC=1 W7_NODES=1 W7_LOCAL_WORLD=4 W7_BATCH=8 W7_K=4 \
  W7_CTX_TOKENS=2048 W7_MAX_MODEL_LEN=6144 W7_MAX_NUM_BATCHED=8192 \
  W7_TRACE_LEN=40 W7_MASTER_PORT=15740 W7_TRACE_DIR="$TRACE_DIR" \
  W7_PROMPT_FILE="$PHASE/../57_large_ep_spec_strategy/data/prompts_ondist.txt"
W7_NODE_RANK=0 timeout 900 "$PY" \
  /data/smcho/self-spec-moe/research/64_window_e2e/scripts/w7_trace16k.py > "$LOG" 2>&1
echo "nospec trace exit=$? ($(date +%H:%M:%S))"
grep -hE "TRACE16K|Traceback|Error|Exception" "$LOG" | tail -6
ls -lh "$TRACE_DIR" 2>/dev/null | tail -2
