#!/bin/bash
# Phase 65: torch-profiler trace of the W512 self-draft decode cycle at 16k
# on the 2-node fabric, per fix STAGE (base|f1|f12|f123). Reuses the Phase-64
# trace harness (w7_trace16k.py) verbatim; only the stage env rail differs.
# Usage: run_trace16k.sh {base|f1|f12|f123} [K=2] [batch=8] [trace_len=60] \
#          [try_to_s=1500]
set -u
STAGE="${1:?base|f1|f12|f123}"
K="${2:-2}"
BATCH="${3:-8}"
TRACE_LEN="${4:-60}"
TRY_TO="${5:-1500}"
PHASE=/h/v-sukmincho/self-spec-moe/research/65_draft_overhead_opt
P64=/h/v-sukmincho/self-spec-moe/research/64_window_e2e
PY=/h/v-sukmincho/self-spec-moe/.venv/bin/python
ENV="$PHASE/scripts/env_selfspec_2node.sh"
TRACE_DIR="$PHASE/data/trace_${STAGE}_w512k${K}_b${BATCH}"
mkdir -p "$PHASE/logs" "$PHASE/data"

case "$STAGE" in
  base) FIX="" ;;
  f1)   FIX="VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1" ;;
  f12)  FIX="VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 \
VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1" ;;
  f123) FIX="VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 \
VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1" ;;
  *) echo "unknown stage $STAGE"; exit 2 ;;
esac
QUANT="W7_DRAFT_QUANT= W7_DRAFT_FULL_REPLICA=0 W7_DRAFT_LOCAL_ROUTE=0"
[ "$STAGE" = "f123" ] && QUANT="W7_DRAFT_QUANT=fp8 W7_DRAFT_FULL_REPLICA=1 \
W7_DRAFT_LOCAL_ROUTE=1"

kill_stragglers() {
  for pat in 'w7_trace16k[.]py' 'w7_2node[.]py' 'EngineCor[e]' 'Worker_D[P]'; do
    pkill -9 -f "$pat" 2>/dev/null
    ssh h106 "pkill -9 -f '$pat'" 2>/dev/null
  done
  sleep 8
}

foreign_gpu_busy() {
  local me p u line
  me="$(whoami)"
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader \
               2>/dev/null); do
    u="$(ps -o user= -p "$p" 2>/dev/null | tr -d ' ')"
    if [ -n "$u" ] && [ "$u" != "$me" ]; then
      echo "[p65-trace] foreign GPU pid=$p user=$u on h107"; return 0
    fi
  done
  while read -r line; do
    p="${line%% *}"; u="${line##* }"
    if [ -n "$u" ] && [ "$u" != "$p" ] && [ "$u" != "$me" ]; then
      echo "[p65-trace] foreign GPU pid=$p user=$u on h106"; return 0
    fi
  done < <(ssh h106 'for p in $(nvidia-smi --query-compute-apps=pid \
             --format=csv,noheader 2>/dev/null); do \
             u=$(ps -o user= -p $p 2>/dev/null | tr -d " "); \
             [ -n "$u" ] && echo "$p $u"; done' 2>/dev/null)
  return 1
}

while foreign_gpu_busy; do
  echo "[p65-trace] waiting 120s for foreign GPU work ($(date +%H:%M:%S))"
  sleep 120
done
kill_stragglers

RUN="W7_K=$K W7_BATCH=$BATCH W7_TRACE_LEN=$TRACE_LEN W7_TRACE_DIR=$TRACE_DIR \
W7_CTX_TOKENS=16384 W7_MAX_MODEL_LEN=20480 W7_MAX_NUM_BATCHED=8192 \
W7_GPU_MEM=0.90 W7_MASTER_PORT=15500 VLLM_CUSTOM_SCOPES_FOR_PROFILING=1 \
$QUANT $FIX W7_DRAFT_NODE_LOCAL=0 VLLM_SELF_SPEC_DRAFT_KV_WINDOW=512 \
VLLM_SELF_SPEC_DRAFT_KV_SINKS=16 W7_KV_WINDOW_DEBUG=1"
LOGH="$PHASE/logs/trace_${STAGE}_w512k${K}_b${BATCH}.log"
LOG6="$PHASE/logs/trace_${STAGE}_w512k${K}_b${BATCH}_h106.log"
echo "[p65-trace] stage=$STAGE K=$K b=$BATCH len=$TRACE_LEN to=${TRY_TO}s ($(date +%H:%M:%S))"
ssh h106 "bash -c 'source $ENV && export $RUN && W7_NODE_RANK=1 timeout $TRY_TO $PY $P64/scripts/w7_trace16k.py'" \
    > "$LOG6" 2>&1 &
H6=$!
( source "$ENV" && export $RUN && W7_NODE_RANK=0 timeout "$TRY_TO" $PY "$P64/scripts/w7_trace16k.py" ) \
    > "$LOGH" 2>&1
RC=$?
for _ in $(seq 24); do kill -0 "$H6" 2>/dev/null || break; sleep 5; done
kill -9 "$H6" 2>/dev/null
wait "$H6" 2>/dev/null
kill_stragglers
grep -hE "TRACE16K|kv-window" "$LOGH" | tail -4
ls -lh "$TRACE_DIR" 2>/dev/null | tail -3
echo "[p65-trace] done RC=$RC ($(date +%H:%M:%S))"
