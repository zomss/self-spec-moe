#!/bin/bash
# Phase 63: 8-rail 2-node DeepSeek-V2 236B at 16k CONTEXT (MLA long-context
# probe). Clone of research/61_236b_one_rail/scripts/run_8rail_236b.sh with
# ONLY these changes: W7_CTX_TOKENS=16384, W7_MAX_MODEL_LEN=20480, tag
# dsv2_8rail_16k, port 14100+, default batches 8,16,32, TRY_TO 900, 2 tries.
# Everything else identical to the Phase 61 1k baselines (synthetic non-chat
# prompts, gpu_mem 0.93, CG sizes, mnb 2048) -- comparability is the point.
# Usage: run_16k_236b.sh {nospec|worldA} [batches] [retries] [try_to_s]
# Env: NCCLDBG=1 adds NCCL_DEBUG=INFO.
set -u
ARM="${1:?usage: run_16k_236b.sh nospec|worldA [batches] [retries] [try_to_s]}"
BATCHES="${2:-8,16,32}"
RETRIES="${3:-2}"
TRY_TO="${4:-900}"
PHASE=/h/v-sukmincho/self-spec-moe/research/63_mla_context
P52=/h/v-sukmincho/self-spec-moe/research/52_two_node_e2e
PY=/h/v-sukmincho/self-spec-moe/.venv/bin/python
ENV="$PHASE/scripts/env_8rail_2node.sh"
mkdir -p "$PHASE/logs" "$PHASE/data"

# Phase 53/54 known-good 236B engine settings; default synthetic prompts
# (NO W7_PROMPT_FILE/W7_CHAT) to match the Phase 61 8-rail 1k references.
# Deltas vs Phase 61: W7_CTX_TOKENS=16384 (pads each synthetic prompt to
# ~16.3k tokens, distinct per request) + W7_MAX_MODEL_LEN=20480.
COMMON="W7_MODEL=deepseek-ai/DeepSeek-V2 W7_TRC=1 \
W7_NODES=2 W7_LOCAL_WORLD=4 W7_TP=2 W7_ITERS=2 W7_WARMUP=1 \
W7_GPU_MEM=0.93 W7_CTX_TOKENS=16384 W7_MAX_MODEL_LEN=20480 \
W7_CG_SIZES=8,16,32,64,128 W7_MAX_NUM_BATCHED=2048 W7_OUT=$PHASE/data"

case "$ARM" in
  nospec)
    MODE=nospec; K=0; TAG=dsv2_8rail_16k; PORT=14100
    OVR="$COMMON" ;;
  worldA)
    MODE=spec; K=1; TAG=dsv2_8rail_16k_worldA; PORT=14150
    OVR="$COMMON W7_KS=1 W7_DRAFT_LOCAL_ROUTE=0 W7_DRAFT_NODE_LOCAL=1 \
W7_DRAFT_FULL_REPLICA=0 VLLM_SELF_SPEC_DRAFT_SKIP_DP_COORD=0 \
VLLM_SELF_SPEC_DRAFT_AMORTIZE_DP_COORD=0 \
VLLM_SELF_SPEC_DRAFT_STEP0_FULL_CG=0" ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac

kill_stragglers() {
  for pat in 'w7_2node[.]py' 'EngineCor[e]' 'Worker_D[P]'; do
    pkill -9 -f "$pat" 2>/dev/null
    ssh h106 "pkill -9 -f '$pat'" 2>/dev/null
  done
  # Shared nodes -- cleanup is pkill of OUR patterns only (own-user).
  # Never kill arbitrary GPU PIDs / other users' jobs.
  sleep 8
}

if [ "$MODE" = "nospec" ]; then
  ROWPAT="W7-2N nospec\] batch=.* tok/s"
else
  ROWPAT="W7-2N K=$K\] batch=.* accept_len"
fi
nb=$(echo "$BATCHES" | awk -F, '{print NF}')

ok=0
for try in $(seq 1 "$RETRIES"); do
  kill_stragglers
  LOGH="$PHASE/logs/${ARM}_try${try}.log"
  LOG6="$PHASE/logs/${ARM}_try${try}_h106.log"
  RUN="$OVR W7_BATCHES=$BATCHES W7_TAG=$TAG \
W7_MASTER_PORT=$((PORT+try*7)) ${NCCLDBG:+NCCL_DEBUG=INFO}"
  echo "[16k-236b] $ARM try=$try to=${TRY_TO}s ($(date +%H:%M:%S))"
  ssh h106 "bash -c 'source $ENV && export $RUN && W7_NODE_RANK=1 timeout $TRY_TO $PY $P52/scripts/w7_2node.py $MODE'" \
      > "$LOG6" 2>&1 &
  H6=$!
  ( source "$ENV" && export $RUN && W7_NODE_RANK=0 timeout "$TRY_TO" $PY "$P52/scripts/w7_2node.py" "$MODE" ) \
      > "$LOGH" 2>&1
  # Bounded wait: orphaned remote engine children can hold the ssh pipe open
  # past the remote timeout (observed: try killed at 900s -> engines idle on
  # both nodes, ssh never exits, retry loop wedges). Give 120s, then kill.
  for _ in $(seq 24); do kill -0 "$H6" 2>/dev/null || break; sleep 5; done
  kill -9 "$H6" 2>/dev/null
  wait "$H6" 2>/dev/null
  got=$(grep -cE "$ROWPAT" "$LOGH" 2>/dev/null)
  echo "[16k-236b] $ARM try=$try -> ${got:-0}/$nb batch rows ($(date +%H:%M:%S))"
  grep -hE "W7-2N" "$LOGH" 2>/dev/null
  if [ "${got:-0}" -ge "$nb" ]; then ok=1; break; fi
done
[ "$ok" = "1" ] || echo "[16k-236b] $ARM INCOMPLETE after $RETRIES tries"
kill_stragglers
echo "[16k-236b] $ARM done ($(date +%H:%M:%S))"
