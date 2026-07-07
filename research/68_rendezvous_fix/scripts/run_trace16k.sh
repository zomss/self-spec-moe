#!/bin/bash
# Phase 68: torch-profiler trace of the arm-B (fp8 full-replica comm-free
# draft + shared-KV + W512 window + Phase-65 flag stack) self-draft decode
# cycle at 16k on the 2-node fabric h107(rank0) + ${W7_PEER:-h108}(rank1).
# The decisive attribution measurement: how many cross-node NCCL collectives
# fire per cycle, and which sit on the draft-step path vs the verify path.
# Clone of research/65_draft_overhead_opt/scripts/run_trace16k.sh with the
# hardcoded `ssh h106` replaced by `ssh ${W7_PEER:-h108}` (peer spawn / pkill
# / foreign-GPU check / log suffix) and the arm-B env baked in.
# Usage: run_trace16k.sh [K=4] [batch=12] [trace_len=80] [try_to_s=1800] [tag]
set -u
K="${1:-4}"
BATCH="${2:-12}"
TRACE_LEN="${3:-80}"
TRY_TO="${4:-1800}"
TAG="${5:-armb}"
PEER="${W7_PEER:-h108}"
PHASE=/h/v-sukmincho/self-spec-moe/research/68_rendezvous_fix
P65=/h/v-sukmincho/self-spec-moe/research/65_draft_overhead_opt
PY=/h/v-sukmincho/self-spec-moe/.venv/bin/python
ENV="$PHASE/scripts/env_selfspec_2node.sh"
TRACE_DIR="$PHASE/data/trace_${TAG}_w512k${K}_b${BATCH}"
mkdir -p "$PHASE/logs" "$PHASE/data"

# Arm-B: fp8 full-replica comm-free draft + shared KV + W512 + P65 flags.
# TRACE_EXTRA lets the caller add the Phase-68 fix flag (default off).
ARMB="W7_DRAFT_QUANT=fp8 W7_DRAFT_FULL_REPLICA=1 W7_DRAFT_LOCAL_ROUTE=1 \
W7_DRAFT_NODE_LOCAL=0 VLLM_SELF_SPEC_DRAFT_SKIP_DP_COORD=1 \
VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1 \
VLLM_SELF_SPEC_SHARED_KV=1 VLLM_SELF_SPEC_DRAFT_KV_WINDOW=512 \
VLLM_SELF_SPEC_DRAFT_KV_SINKS=16 W7_KV_WINDOW_DEBUG=1 ${TRACE_EXTRA:-}"

kill_stragglers() {
  for pat in 'w7_trace16k[.]py' 'w7_2node[.]py' 'EngineCor[e]' 'Worker_D[P]'; do
    pkill -9 -f "$pat" 2>/dev/null
    ssh "$PEER" "pkill -9 -f '$pat'" 2>/dev/null
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
      echo "[p68-trace] foreign GPU pid=$p user=$u on h107"; return 0
    fi
  done
  while read -r line; do
    p="${line%% *}"; u="${line##* }"
    if [ -n "$u" ] && [ "$u" != "$p" ] && [ "$u" != "$me" ]; then
      echo "[p68-trace] foreign GPU pid=$p user=$u on $PEER"; return 0
    fi
  done < <(ssh "$PEER" 'for p in $(nvidia-smi --query-compute-apps=pid \
             --format=csv,noheader 2>/dev/null); do \
             u=$(ps -o user= -p $p 2>/dev/null | tr -d " "); \
             [ -n "$u" ] && echo "$p $u"; done' 2>/dev/null)
  return 1
}

while foreign_gpu_busy; do
  echo "[p68-trace] waiting 120s for foreign GPU work ($(date +%H:%M:%S))"
  sleep 120
done
kill_stragglers

RUN="W7_K=$K W7_BATCH=$BATCH W7_TRACE_LEN=$TRACE_LEN W7_TRACE_DIR=$TRACE_DIR \
W7_CTX_TOKENS=16384 W7_MAX_MODEL_LEN=20480 W7_MAX_NUM_BATCHED=8192 \
W7_GPU_MEM=0.90 W7_MASTER_PORT=16800 VLLM_CUSTOM_SCOPES_FOR_PROFILING=1 \
$ARMB"
LOGH="$PHASE/logs/trace_${TAG}_w512k${K}_b${BATCH}.log"
LOGP="$PHASE/logs/trace_${TAG}_w512k${K}_b${BATCH}_${PEER}.log"
echo "[p68-trace] arm=$TAG K=$K b=$BATCH len=$TRACE_LEN peer=$PEER to=${TRY_TO}s ($(date +%H:%M:%S))"
ssh "$PEER" "bash -c 'source $ENV && export $RUN && W7_NODE_RANK=1 timeout $TRY_TO $PY $PHASE/scripts/w7_trace16k.py'" \
    > "$LOGP" 2>&1 &
HP=$!
( source "$ENV" && export $RUN && W7_NODE_RANK=0 timeout "$TRY_TO" $PY "$PHASE/scripts/w7_trace16k.py" ) \
    > "$LOGH" 2>&1
RC=$?
for _ in $(seq 24); do kill -0 "$HP" 2>/dev/null || break; sleep 5; done
kill -9 "$HP" 2>/dev/null
wait "$HP" 2>/dev/null
kill_stragglers
grep -hE "TRACE16K|kv-window|shared-KV" "$LOGH" | tail -6
ls -lh "$TRACE_DIR" 2>/dev/null | tail -3
echo "[p68-trace] done RC=$RC ($(date +%H:%M:%S))"
