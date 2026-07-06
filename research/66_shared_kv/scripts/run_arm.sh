#!/bin/bash
# Phase 66: 2-node (h107+h106, DP16/EP16) 16k measurement of the shared-KV
# self-draft at serving batches. Clone of research/65_draft_overhead_opt/
# scripts/run_arm.sh with VLLM_SELF_SPEC_SHARED_KV=1 and per-arm draft
# configs (all arms: W512 sinks=16 + the Phase-65 flag stack).
#   a_ep  : EP-routed bf16 self-draft (zero extra weights; draft pays EP
#           collectives). Fixes: DP_COORD_CPU + CHAIN_LIGHT_MD (NO
#           SKIP_DP_COORD -- the EP-routed draft needs the DP agreement).
#   b_rep : fp8 full-replica comm-free draft (Phase 65 best-D config):
#           + LOCAL_ROUTE + SKIP_DP_COORD (safe only comm-free).
#   c_nl  : node-local draft (W7_DRAFT_NODE_LOCAL=1, no replica, comm
#           halved). DP_COORD_CPU + CHAIN_LIGHT_MD.
#   nospec: no-spec reference (refs stand from Phase 64; only for re-checks).
# Usage: run_arm.sh {nospec|a_ep|b_rep|c_nl} K [batches=12,32] [retries=2] \
#          [try_to=2400] [tag_suffix]
set -u
ARM="${1:?usage: run_arm.sh nospec|a_ep|b_rep|c_nl K [batches] ...}"
K="${2:?K (0 for nospec)}"
BATCHES="${3:-12,32}"
RETRIES="${4:-2}"
TRY_TO="${5:-2400}"
TAGSUF="${6:-}"
PHASE=/h/v-sukmincho/self-spec-moe/research/66_shared_kv
P52=/h/v-sukmincho/self-spec-moe/research/52_two_node_e2e
P57=/h/v-sukmincho/self-spec-moe/research/57_large_ep_spec_strategy
PY=/h/v-sukmincho/self-spec-moe/.venv/bin/python
mkdir -p "$PHASE/logs" "$PHASE/data"

COMMON="W7_NODES=2 W7_LOCAL_WORLD=8 W7_ITERS=2 W7_WARMUP=1 W7_GPU_MEM=0.90 \
W7_CTX_TOKENS=16384 W7_MAX_MODEL_LEN=20480 W7_MAX_NUM_BATCHED=8192 \
W7_PROMPT_FILE=$P57/data/prompts_ondist.txt W7_CHAT=1 W7_OUT=$PHASE/data"

# Phase-65 fixes 1+2 (safe for every draft) + Phase-66 shared KV + window.
FIXBASE="VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 \
VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1 VLLM_SELF_SPEC_SHARED_KV=1 \
VLLM_SELF_SPEC_DRAFT_KV_WINDOW=512 VLLM_SELF_SPEC_DRAFT_KV_SINKS=16 \
W7_KV_WINDOW_DEBUG=1"
ENVSS="$PHASE/scripts/env_selfspec_2node.sh"
ENVEA="$P57/scripts/env_eagle_2node.sh"

case "$ARM" in
  nospec)
    MODE=nospec; ENV=$ENVEA; PORT=15400; TAG=q30b_p66_nospec
    OVR="$COMMON" ;;
  a_ep)
    MODE=spec; ENV=$ENVSS; PORT=15450; TAG=q30b_p66_aep
    OVR="$COMMON $FIXBASE W7_DRAFT_QUANT= W7_DRAFT_FULL_REPLICA=0 \
W7_DRAFT_LOCAL_ROUTE=0 W7_DRAFT_NODE_LOCAL=0 W7_KS=$K" ;;
  b_rep)
    MODE=spec; ENV=$ENVSS; PORT=15500; TAG=q30b_p66_brep
    OVR="$COMMON $FIXBASE W7_DRAFT_QUANT=fp8 W7_DRAFT_FULL_REPLICA=1 \
W7_DRAFT_LOCAL_ROUTE=1 W7_DRAFT_NODE_LOCAL=0 \
VLLM_SELF_SPEC_DRAFT_SKIP_DP_COORD=1 W7_KS=$K" ;;
  c_nl)
    MODE=spec; ENV=$ENVSS; PORT=15550; TAG=q30b_p66_cnl
    OVR="$COMMON $FIXBASE W7_DRAFT_QUANT= W7_DRAFT_FULL_REPLICA=0 \
W7_DRAFT_LOCAL_ROUTE=0 W7_DRAFT_NODE_LOCAL=1 W7_KS=$K" ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac
OVR="$OVR ${P66_EXTRA:-}"
TAG="$TAG$TAGSUF"

kill_stragglers() {
  # Shared nodes -- cleanup is pkill of OUR bracket-escaped patterns only
  # (own-user procs). Never kill arbitrary GPU PIDs, never sudo.
  for pat in 'w7_2node[.]py' 'EngineCor[e]' 'Worker_D[P]'; do
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
      echo "[p66] foreign GPU pid=$p user=$u on h107"; return 0
    fi
  done
  while read -r line; do
    p="${line%% *}"; u="${line##* }"
    if [ -n "$u" ] && [ "$u" != "$p" ] && [ "$u" != "$me" ]; then
      echo "[p66] foreign GPU pid=$p user=$u on h106"; return 0
    fi
  done < <(ssh h106 'for p in $(nvidia-smi --query-compute-apps=pid \
             --format=csv,noheader 2>/dev/null); do \
             u=$(ps -o user= -p $p 2>/dev/null | tr -d " "); \
             [ -n "$u" ] && echo "$p $u"; done' 2>/dev/null)
  return 1
}

if [ "$MODE" = "nospec" ]; then
  ROWPAT="W7-2N nospec\] batch=.* tok/s"
else
  ROWPAT="W7-2N K=$K\] batch=.* accept_len"
fi
nb=$(echo "$BATCHES" | awk -F, '{print NF}')

ok=0
for try in $(seq 1 "$RETRIES"); do
  while foreign_gpu_busy; do
    echo "[p66] $ARM K=$K: waiting 120s for foreign GPU work ($(date +%H:%M:%S))"
    sleep 120
  done
  kill_stragglers
  LOGH="$PHASE/logs/${ARM}_K${K}${TAGSUF}_try${try}.log"
  LOG6="$PHASE/logs/${ARM}_K${K}${TAGSUF}_try${try}_h106.log"
  RUN="$OVR W7_BATCHES=$BATCHES W7_TAG=$TAG \
W7_MASTER_PORT=$((PORT+try*7)) ${NCCLDBG:+NCCL_DEBUG=INFO}"
  echo "[p66] $ARM K=$K b=$BATCHES try=$try to=${TRY_TO}s ($(date +%H:%M:%S))"
  ssh h106 "bash -c 'source $ENV && export $RUN && W7_NODE_RANK=1 timeout $TRY_TO $PY $P52/scripts/w7_2node.py $MODE'" \
      > "$LOG6" 2>&1 &
  H6=$!
  ( source "$ENV" && export $RUN && W7_NODE_RANK=0 timeout "$TRY_TO" $PY "$P52/scripts/w7_2node.py" "$MODE" ) \
      > "$LOGH" 2>&1
  for _ in $(seq 24); do kill -0 "$H6" 2>/dev/null || break; sleep 5; done
  kill -9 "$H6" 2>/dev/null
  wait "$H6" 2>/dev/null
  got=$(grep -cE "$ROWPAT" "$LOGH" 2>/dev/null)
  echo "[p66] $ARM K=$K try=$try -> ${got:-0}/$nb batch rows ($(date +%H:%M:%S))"
  grep -hE "W7-2N|kv-window|KV cache size|shared-KV" "$LOGH" 2>/dev/null | tail -12
  if [ "${got:-0}" -ge "$nb" ]; then ok=1; break; fi
done
kill_stragglers
[ "$ok" = "1" ] || { echo "[p66] $ARM K=$K INCOMPLETE after $RETRIES tries"; exit 1; }
echo "[p66] $ARM K=$K done ($(date +%H:%M:%S))"
