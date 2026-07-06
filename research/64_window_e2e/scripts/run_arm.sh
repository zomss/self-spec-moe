#!/bin/bash
# Phase 64: END-TO-END serving tok/s of the window-KV self-draft at 16k on the
# real 2-node EP16 fabric (h107+h106). Clone of the FIXED Phase 63 runner
# (research/63_mla_context/scripts/run_16k_236b.sh, commit b8246b860):
# hard per-try `timeout` on BOTH sides + BOUNDED wait on the h106 ssh
# (orphaned engine children hold the pipe open past the remote timeout and
# wedge the retry loop) + own-pattern straggler kill on both nodes + retry
# with a bumped master port. Adds the Phase-62 foreign-GPU guard, on BOTH
# nodes (shared nodes: never contend, never kill others' PIDs).
# Usage: run_arm.sh {nospec|w512k2|w512k4|w256k4|eagle_k1} \
#          [batches=8,32] [retries=2] [try_to_s=1200] [tag_suffix]
# tag_suffix keeps split invocations of the same arm (e.g. the over-pool b32
# point, which needs its own long-timeout engine launch) in separate JSONs
# and logs.
# Env: NCCLDBG=1 adds NCCL_DEBUG=INFO.
set -u
ARM="${1:?usage: run_arm.sh nospec|w512k2|w512k4|w256k4|eagle_k1}"
BATCHES="${2:-8,32}"
RETRIES="${3:-2}"
TRY_TO="${4:-1200}"
TAGSUF="${5:-}"
PHASE=/h/v-sukmincho/self-spec-moe/research/64_window_e2e
P52=/h/v-sukmincho/self-spec-moe/research/52_two_node_e2e
P57=/h/v-sukmincho/self-spec-moe/research/57_large_ep_spec_strategy
PY=/h/v-sukmincho/self-spec-moe/.venv/bin/python
mkdir -p "$PHASE/logs" "$PHASE/data"

# Phase 59/62 16k conventions: chat + on-dist prompts padded to ~16.4k,
# iters=2 warmup=1, MNB=8192, gpu_mem 0.90. Batches 8,32 (do NOT run b64).
COMMON="W7_NODES=2 W7_LOCAL_WORLD=8 W7_ITERS=2 W7_WARMUP=1 W7_GPU_MEM=0.90 \
W7_CTX_TOKENS=16384 W7_MAX_MODEL_LEN=20480 W7_MAX_NUM_BATCHED=8192 \
W7_PROMPT_FILE=$P57/data/prompts_ondist.txt W7_CHAT=1 W7_OUT=$PHASE/data"

# Arms 2-4: EP-routed bf16 self-draft -- NO quant, NO replica, NO local/node
# routing (mission spec; harness defaults LOCAL_ROUTE=1/FULL_REPLICA=1, so
# every one must be exported explicitly). Window engagement evidence via
# W7_KV_WINDOW_DEBUG ([kv-window] lines, 1 per 500 draft builds).
SELFSPEC="W7_DRAFT_QUANT= W7_DRAFT_FULL_REPLICA=0 W7_DRAFT_LOCAL_ROUTE=0 \
W7_DRAFT_NODE_LOCAL=0 VLLM_SELF_SPEC_DRAFT_KV_SINKS=16 W7_KV_WINDOW_DEBUG=1"
ENVSS="$PHASE/scripts/env_selfspec_2node.sh"
ENVEA="$P57/scripts/env_eagle_2node.sh"

case "$ARM" in
  nospec)   # fresh same-session denominator (Phase 59 env, ref 348.5/367.9)
    MODE=nospec; K=0; ENV=$ENVEA; PORT=14200; TAG=q30b_p64_nospec
    OVR="$COMMON" ;;
  w512k2)
    MODE=spec; K=2; ENV=$ENVSS; PORT=14250; TAG=q30b_p64_w512
    OVR="$COMMON $SELFSPEC W7_KS=2 VLLM_SELF_SPEC_DRAFT_KV_WINDOW=512" ;;
  w512k4)   # the headline arm
    MODE=spec; K=4; ENV=$ENVSS; PORT=14300; TAG=q30b_p64_w512
    OVR="$COMMON $SELFSPEC W7_KS=4 VLLM_SELF_SPEC_DRAFT_KV_WINDOW=512" ;;
  w256k4)
    MODE=spec; K=4; ENV=$ENVSS; PORT=14350; TAG=q30b_p64_w256
    OVR="$COMMON $SELFSPEC W7_KS=4 VLLM_SELF_SPEC_DRAFT_KV_WINDOW=256" ;;
  eagle_k1) # optional re-reference (Phase 59 env stack, real EAGLE3 head)
    MODE=spec; K=1; ENV=$ENVEA; PORT=14400; TAG=q30b_p64_eagle
    OVR="$COMMON W7_KS=1" ;;
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
  # Any GPU compute PID owned by ANOTHER user on EITHER node -> busy.
  # PIDs with no live process (h106's stale 522 MiB context) resolve to an
  # empty user and are skipped.
  local me p u line
  me="$(whoami)"
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader \
               2>/dev/null); do
    u="$(ps -o user= -p "$p" 2>/dev/null | tr -d ' ')"
    if [ -n "$u" ] && [ "$u" != "$me" ]; then
      echo "[p64] foreign GPU pid=$p user=$u on h107"; return 0
    fi
  done
  while read -r line; do
    p="${line%% *}"; u="${line##* }"
    if [ -n "$u" ] && [ "$u" != "$p" ] && [ "$u" != "$me" ]; then
      echo "[p64] foreign GPU pid=$p user=$u on h106"; return 0
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
    echo "[p64] $ARM: waiting 120s for foreign GPU work ($(date +%H:%M:%S))"
    sleep 120
  done
  kill_stragglers
  LOGH="$PHASE/logs/${ARM}${TAGSUF}_try${try}.log"
  LOG6="$PHASE/logs/${ARM}${TAGSUF}_try${try}_h106.log"
  RUN="$OVR W7_BATCHES=$BATCHES W7_TAG=$TAG \
W7_MASTER_PORT=$((PORT+try*7)) ${NCCLDBG:+NCCL_DEBUG=INFO}"
  echo "[p64] $ARM try=$try to=${TRY_TO}s ($(date +%H:%M:%S))"
  ssh h106 "bash -c 'source $ENV && export $RUN && W7_NODE_RANK=1 timeout $TRY_TO $PY $P52/scripts/w7_2node.py $MODE'" \
      > "$LOG6" 2>&1 &
  H6=$!
  ( source "$ENV" && export $RUN && W7_NODE_RANK=0 timeout "$TRY_TO" $PY "$P52/scripts/w7_2node.py" "$MODE" ) \
      > "$LOGH" 2>&1
  # Bounded wait (Phase 63 fix): orphaned remote engine children can hold the
  # ssh pipe open past the remote timeout and wedge the retry loop. 120s max.
  for _ in $(seq 24); do kill -0 "$H6" 2>/dev/null || break; sleep 5; done
  kill -9 "$H6" 2>/dev/null
  wait "$H6" 2>/dev/null
  got=$(grep -cE "$ROWPAT" "$LOGH" 2>/dev/null)
  echo "[p64] $ARM try=$try -> ${got:-0}/$nb batch rows ($(date +%H:%M:%S))"
  grep -hE "W7-2N|kv-window|KV cache size" "$LOGH" 2>/dev/null | tail -12
  if [ "${got:-0}" -ge "$nb" ]; then ok=1; break; fi
done
kill_stragglers
[ "$ok" = "1" ] || { echo "[p64] $ARM INCOMPLETE after $RETRIES tries"; exit 1; }
echo "[p64] $ARM done ($(date +%H:%M:%S))"
