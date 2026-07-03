#!/bin/bash
# Phase 49 profiling pass: decompose the 2-node spec cycle (Phase-44 recon
# methodology). Spec mode only, b64, 1 warmup + 1 iter, PROFILE=1.
set -u
PHASE=/h/v-sukmincho/self-spec-moe/research/52_two_node_e2e
PY=/h/v-sukmincho/self-spec-moe/.venv/bin/python
mkdir -p "$PHASE/logs" "$PHASE/data/profile"

OVR="W7_BATCHES=64 W7_ITERS=1 W7_WARMUP=1 W7_TAG=qwen30b_2node_prof \
VLLM_SELF_SPEC_PROFILE=1 VLLM_SELF_SPEC_PROFILE_FINE=1 \
VLLM_SELF_SPEC_PROFILE_OUT=$PHASE/data/profile"

kill_stragglers() {
  pkill -9 -f "w7_2node" 2>/dev/null
  pkill -9 -f "EngineCore" 2>/dev/null
  ssh h106 'pkill -9 -f w7_2node; pkill -9 -f EngineCore' 2>/dev/null
  sleep 5
}

kill_stragglers

ssh h106 "bash -c 'source $PHASE/scripts/env_2node.sh && export $OVR && W7_NODE_RANK=1 exec $PY $PHASE/scripts/w7_2node.py spec'" \
    > "$PHASE/logs/prof_h106.log" 2>&1 &
H106_PID=$!

source "$PHASE/scripts/env_2node.sh"
export $OVR
W7_NODE_RANK=0 $PY "$PHASE/scripts/w7_2node.py" spec \
    > "$PHASE/logs/prof_h107.log" 2>&1
RC=$?
wait "$H106_PID"
kill_stragglers
echo "run_2node_profile done RC=$RC"
exit $RC
