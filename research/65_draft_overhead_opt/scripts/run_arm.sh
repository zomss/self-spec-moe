#!/bin/bash
# Phase 65: stage-by-stage measurement of the draft-path overhead fixes at
# 16k on the real 2-node EP16 fabric (h107+h106). Clone of the Phase 64
# runner (research/64_window_e2e/scripts/run_arm.sh) with a STAGE argument
# that layers the fix flags:
#   base : Phase-64 "before" config (EP-routed bf16 self-draft, no fixes)
#   f1   : + VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1        (Fix 1: no GPU syncs)
#   f12  : f1 + VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1   (Fix 2: chain CPU)
#   f123 : f12 + fp8 FULL-REPLICA comm-free draft       (Fix 3: no draft NCCL)
# Usage: run_arm.sh {nospec|w512k2|w512k4} {base|f1|f12|f123} \
#          [batches=8,12] [retries=2] [try_to=2400] [tag_suffix]
# Protocol (Phase 64): 16k chat+on-dist prompts, MNB 8192, gpu_mem 0.90,
# iters=2 warmup=1, two-length slope 160/32. b32 remains pool-blocked: DO NOT
# run it (2.3x over the self-spec KV pool -> preemption livelock).
set -u
ARM="${1:?usage: run_arm.sh nospec|w512k2|w512k4 base|f1|f12|f123}"
STAGE="${2:?stage: base|f1|f12|f123}"
BATCHES="${3:-8,12}"
RETRIES="${4:-2}"
TRY_TO="${5:-2400}"
TAGSUF="${6:-}"
PHASE=/h/v-sukmincho/self-spec-moe/research/65_draft_overhead_opt
P52=/h/v-sukmincho/self-spec-moe/research/52_two_node_e2e
P57=/h/v-sukmincho/self-spec-moe/research/57_large_ep_spec_strategy
PY=/h/v-sukmincho/self-spec-moe/.venv/bin/python
mkdir -p "$PHASE/logs" "$PHASE/data"

COMMON="W7_NODES=2 W7_LOCAL_WORLD=8 W7_ITERS=2 W7_WARMUP=1 W7_GPU_MEM=0.90 \
W7_CTX_TOKENS=16384 W7_MAX_MODEL_LEN=20480 W7_MAX_NUM_BATCHED=8192 \
W7_PROMPT_FILE=$P57/data/prompts_ondist.txt W7_CHAT=1 W7_OUT=$PHASE/data"

# EP-routed bf16 self-draft (Phase 64 "before" arms). Stage f123 switches the
# DRAFT to the banked Phase-52 fp8 FULL-REPLICA comm-free path (accept with
# the window already measured: 4.547 @K4 W512, Phase 62 arm C).
SELFSPEC_EP="W7_DRAFT_QUANT= W7_DRAFT_FULL_REPLICA=0 W7_DRAFT_LOCAL_ROUTE=0 \
W7_DRAFT_NODE_LOCAL=0 VLLM_SELF_SPEC_DRAFT_KV_SINKS=16 W7_KV_WINDOW_DEBUG=1"
SELFSPEC_REP="W7_DRAFT_QUANT=fp8 W7_DRAFT_FULL_REPLICA=1 W7_DRAFT_LOCAL_ROUTE=1 \
W7_DRAFT_NODE_LOCAL=0 VLLM_SELF_SPEC_DRAFT_KV_SINKS=16 W7_KV_WINDOW_DEBUG=1"
ENVSS="$PHASE/scripts/env_selfspec_2node.sh"
ENVEA="$P57/scripts/env_eagle_2node.sh"

case "$STAGE" in
  base) FIX="" ;;
  f1)   FIX="VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1" ;;
  f12)  FIX="VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 \
VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1" ;;
  f123) FIX="VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 \
VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1" ;;
  *) echo "unknown stage $STAGE"; exit 2 ;;
esac
SELFSPEC="$SELFSPEC_EP"
[ "$STAGE" = "f123" ] && SELFSPEC="$SELFSPEC_REP"

case "$ARM" in
  nospec)
    MODE=nospec; K=0; ENV=$ENVEA; PORT=15200; TAG=q30b_p65_${STAGE}_nospec
    OVR="$COMMON" ;;
  w512k2)
    MODE=spec; K=2; ENV=$ENVSS; PORT=15250; TAG=q30b_p65_${STAGE}_w512
    OVR="$COMMON $SELFSPEC $FIX W7_KS=2 VLLM_SELF_SPEC_DRAFT_KV_WINDOW=512" ;;
  w512k4)
    MODE=spec; K=4; ENV=$ENVSS; PORT=15300; TAG=q30b_p65_${STAGE}_w512
    OVR="$COMMON $SELFSPEC $FIX W7_KS=4 VLLM_SELF_SPEC_DRAFT_KV_WINDOW=512" ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac
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
      echo "[p65] foreign GPU pid=$p user=$u on h107"; return 0
    fi
  done
  while read -r line; do
    p="${line%% *}"; u="${line##* }"
    if [ -n "$u" ] && [ "$u" != "$p" ] && [ "$u" != "$me" ]; then
      echo "[p65] foreign GPU pid=$p user=$u on h106"; return 0
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
    echo "[p65] $ARM/$STAGE: waiting 120s for foreign GPU work ($(date +%H:%M:%S))"
    sleep 120
  done
  kill_stragglers
  LOGH="$PHASE/logs/${ARM}_${STAGE}${TAGSUF}_try${try}.log"
  LOG6="$PHASE/logs/${ARM}_${STAGE}${TAGSUF}_try${try}_h106.log"
  RUN="$OVR W7_BATCHES=$BATCHES W7_TAG=$TAG \
W7_MASTER_PORT=$((PORT+try*7)) ${NCCLDBG:+NCCL_DEBUG=INFO}"
  echo "[p65] $ARM/$STAGE try=$try to=${TRY_TO}s ($(date +%H:%M:%S))"
  ssh h106 "bash -c 'source $ENV && export $RUN && W7_NODE_RANK=1 timeout $TRY_TO $PY $P52/scripts/w7_2node.py $MODE'" \
      > "$LOG6" 2>&1 &
  H6=$!
  ( source "$ENV" && export $RUN && W7_NODE_RANK=0 timeout "$TRY_TO" $PY "$P52/scripts/w7_2node.py" "$MODE" ) \
      > "$LOGH" 2>&1
  for _ in $(seq 24); do kill -0 "$H6" 2>/dev/null || break; sleep 5; done
  kill -9 "$H6" 2>/dev/null
  wait "$H6" 2>/dev/null
  got=$(grep -cE "$ROWPAT" "$LOGH" 2>/dev/null)
  echo "[p65] $ARM/$STAGE try=$try -> ${got:-0}/$nb batch rows ($(date +%H:%M:%S))"
  grep -hE "W7-2N|kv-window|KV cache size" "$LOGH" 2>/dev/null | tail -12
  if [ "${got:-0}" -ge "$nb" ]; then ok=1; break; fi
done
kill_stragglers
[ "$ok" = "1" ] || { echo "[p65] $ARM/$STAGE INCOMPLETE after $RETRIES tries"; exit 1; }
echo "[p65] $ARM/$STAGE done ($(date +%H:%M:%S))"
