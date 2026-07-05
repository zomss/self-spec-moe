#!/bin/bash
# Phase 57: robust per-K EAGLE K-sweep on real fabric.
# ONE K per engine invocation, each wrapped in a hard `timeout` so the
# intermittent DP16 first-collective wedge is force-killed and retried instead
# of hanging on the 90-min queue timeout. GPU-pid force-kill between tries
# (wedged NCCL workers survive pkill). Per-K JSONs persist under phase data/.
# Usage: run_ksweep.sh {1n|2n} "K1 K2 ..." "b1,b2,.." [tagsfx] [retries] [try_to_s]
set -u
NMODE="${1:?usage: run_ksweep.sh 1n|2n \"K list\" \"batches\" [tagsfx] [retries] [try_to_s]}"
KLIST="${2:?K list}"
BATCHES="${3:-8,32,64}"
TAGSFX="${4:-}"
RETRIES="${5:-4}"
TRY_TO="${6:-540}"
PHASE=/h/v-sukmincho/self-spec-moe/research/57_large_ep_spec_strategy
P52=/h/v-sukmincho/self-spec-moe/research/52_two_node_e2e
PY=/h/v-sukmincho/self-spec-moe/.venv/bin/python
ENV="$PHASE/scripts/env_eagle_2node.sh"
mkdir -p "$PHASE/logs" "$PHASE/data"

if [ "$NMODE" = "1n" ]; then
  NODES=1; TWO_NODE=0; TAG="q30b_eagle_1n${TAGSFX}"; PORT=13500
else
  NODES=2; TWO_NODE=1; TAG="q30b_eagle_2n${TAGSFX}"; PORT=13540
fi

kill_stragglers() {
  for pat in 'w7_2node[.]py' 'EngineCor[e]' 'Worker_D[P]' 'VLLM[:]:'; do
    pkill -9 -f "$pat" 2>/dev/null
    ssh h106 "pkill -9 -f '$pat'" 2>/dev/null
  done
  # wedged NCCL workers ignore pkill: force-kill by GPU pid on both nodes.
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u); do
    kill -9 "$p" 2>/dev/null
  done
  ssh h106 'for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u); do kill -9 "$p" 2>/dev/null; done' 2>/dev/null
  sleep 8
}

nb=$(echo "$BATCHES" | awk -F, '{print NF}')
for K in $KLIST; do
  ok=0
  for try in $(seq 1 "$RETRIES"); do
    kill_stragglers
    LOGH="$PHASE/logs/ksweep_${TAG}_K${K}_try${try}.log"
    LOG6="$PHASE/logs/ksweep_${TAG}_K${K}_try${try}_h106.log"
    OVR="W7_NODES=$NODES W7_LOCAL_WORLD=8 W7_KS=$K W7_BATCHES=$BATCHES \
W7_ITERS=3 W7_WARMUP=2 W7_GPU_MEM=0.90 W7_TAG=$TAG W7_MASTER_PORT=$((PORT+try*20)) ${PROMPTF:+W7_PROMPT_FILE=$PROMPTF} ${CHAT:+W7_CHAT=$CHAT}"
    echo "[ksweep] $TAG K=$K try=$try to=${TRY_TO}s ($(date +%H:%M:%S))"
    if [ "$TWO_NODE" = "1" ]; then
      ssh h106 "bash -c 'source $ENV && export $OVR && W7_NODE_RANK=1 timeout $TRY_TO $PY $P52/scripts/w7_2node.py spec'" \
          > "$LOG6" 2>&1 &
      H6=$!
    fi
    ( source "$ENV" && export $OVR && W7_NODE_RANK=0 timeout "$TRY_TO" $PY "$P52/scripts/w7_2node.py" spec ) \
        > "$LOGH" 2>&1
    [ "$TWO_NODE" = "1" ] && wait "$H6" 2>/dev/null
    got=$(grep -cE "W7-2N K=$K\] batch=.* accept_len" "$LOGH" 2>/dev/null)
    echo "[ksweep] K=$K try=$try -> ${got:-0}/$nb batch rows ($(date +%H:%M:%S))"
    grep -hE "W7-2N K=$K" "$LOGH" 2>/dev/null
    if [ "${got:-0}" -ge "$nb" ]; then ok=1; break; fi
  done
  [ "$ok" = "1" ] || echo "[ksweep] K=$K INCOMPLETE after $RETRIES tries"
done
kill_stragglers
echo "[ksweep] $TAG done ($(date +%H:%M:%S))"
