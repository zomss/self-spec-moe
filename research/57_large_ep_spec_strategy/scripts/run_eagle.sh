#!/bin/bash
# Phase 57: real EAGLE3 head on real fabric, K-sweep at large EP.
# Usage: run_eagle.sh {smoke|sweep1n|sweep2n} [tag_suffix]
#   smoke  : 1-node DP8/EP8, K=2, b8            (de-risk EAGLE on this branch)
#   sweep1n: 1-node DP8/EP8, K-sweep, b{8,32,64} (lower-f K*(EP8) point)
#   sweep2n: 2-node DP16/EP16, K-sweep, b{8,32,64} (higher-f K*(EP16) point)
# Env overrides: W7_KS, W7_BATCHES, W7_ITERS, W7_WARMUP, W7_GPU_MEM, TAGSFX.
set -u
STEP="${1:?usage: run_eagle.sh smoke|sweep1n|sweep2n [tagsfx]}"
TAGSFX="${2:-${TAGSFX:-}}"
PHASE=/h/v-sukmincho/self-spec-moe/research/57_large_ep_spec_strategy
P52=/h/v-sukmincho/self-spec-moe/research/52_two_node_e2e
PY=/h/v-sukmincho/self-spec-moe/.venv/bin/python
ENV="$PHASE/scripts/env_eagle_2node.sh"
mkdir -p "$PHASE/logs" "$PHASE/data"

case "$STEP" in
  smoke)
    OVR="W7_NODES=1 W7_LOCAL_WORLD=8 W7_KS=${W7_KS:-2} W7_BATCHES=${W7_BATCHES:-8} \
W7_ITERS=${W7_ITERS:-2} W7_WARMUP=${W7_WARMUP:-1} W7_GPU_MEM=${W7_GPU_MEM:-0.90} \
W7_TAG=q30b_eagle_1n${TAGSFX} W7_MASTER_PORT=13455"
    TWO_NODE=0 ;;
  sweep1n)
    OVR="W7_NODES=1 W7_LOCAL_WORLD=8 W7_KS=${W7_KS:-1,2,3,4,6,8} \
W7_BATCHES=${W7_BATCHES:-8,32,64} W7_ITERS=${W7_ITERS:-3} W7_WARMUP=${W7_WARMUP:-2} \
W7_GPU_MEM=${W7_GPU_MEM:-0.90} W7_TAG=q30b_eagle_1n${TAGSFX} W7_MASTER_PORT=13460"
    TWO_NODE=0 ;;
  sweep2n)
    OVR="W7_NODES=2 W7_LOCAL_WORLD=8 W7_KS=${W7_KS:-1,2,3,4,6,8} \
W7_BATCHES=${W7_BATCHES:-8,32,64} W7_ITERS=${W7_ITERS:-3} W7_WARMUP=${W7_WARMUP:-2} \
W7_GPU_MEM=${W7_GPU_MEM:-0.90} W7_TAG=q30b_eagle_2n${TAGSFX} W7_MASTER_PORT=13470"
    TWO_NODE=1 ;;
  *) echo "unknown step $STEP"; exit 2 ;;
esac

kill_stragglers() {
  for pat in 'w7_2node[.]py' 'EngineCor[e]' 'Worker_D[P]' 'VLLM[:]:'; do
    pkill -9 -f "$pat" 2>/dev/null
    ssh h106 "pkill -9 -f '$pat'" 2>/dev/null
  done
  sleep 5
}

kill_stragglers

if [ "$TWO_NODE" = "1" ]; then
  ssh h106 "bash -c 'source $ENV && export $OVR && W7_NODE_RANK=1 exec $PY $P52/scripts/w7_2node.py spec'" \
      > "$PHASE/logs/${STEP}${TAGSFX}_h106.log" 2>&1 &
  H106_PID=$!
fi

source "$ENV"
export $OVR
W7_NODE_RANK=0 $PY "$P52/scripts/w7_2node.py" spec \
    > "$PHASE/logs/${STEP}${TAGSFX}_h107.log" 2>&1
RC=$?
[ "$TWO_NODE" = "1" ] && wait "$H106_PID"
kill_stragglers
echo "run $STEP${TAGSFX} done RC=$RC"
grep -h 'W7-2N' "$PHASE/logs/${STEP}${TAGSFX}_h107.log" | tail -20
exit $RC
