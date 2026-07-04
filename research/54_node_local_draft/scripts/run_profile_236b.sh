#!/bin/bash
# Phase 55/Fix-A: profiled 236B node-local spec run (b8 only, region split).
set -u
PHASE=/h/v-sukmincho/self-spec-moe/research/54_node_local_draft
P52=/h/v-sukmincho/self-spec-moe/research/52_two_node_e2e
PY=/h/v-sukmincho/self-spec-moe/.venv/bin/python
mkdir -p "$PHASE/logs" "$PHASE/data/profile236b"

OVR="W7_DRAFT_LOCAL_ROUTE=0 W7_DRAFT_NODE_LOCAL=1 W7_DRAFT_FULL_REPLICA=0 \
W7_KS=1 W7_OUT=$PHASE/data W7_MODEL=deepseek-ai/DeepSeek-V2 W7_TRC=1 \
W7_NODES=2 W7_LOCAL_WORLD=4 W7_TP=2 W7_BATCHES=8 W7_ITERS=1 W7_WARMUP=1 \
W7_GPU_MEM=0.90 W7_MAX_MODEL_LEN=1024 W7_CG_SIZES=8,16,32,64,128 \
W7_MAX_NUM_BATCHED=2048 W7_TAG=prof236b W7_MASTER_PORT=13450 \
VLLM_SELF_SPEC_PROFILE=1 VLLM_SELF_SPEC_PROFILE_OUT=$PHASE/data/profile236b \
VLLM_SELF_SPEC_LOG_A2A_COUNTS=1"

kill_stragglers() {
  for pat in 'w7_2node[.]py' 'EngineCor[e]' 'Worker_D[P]' 'VLLM[:]:'; do
    pkill -9 -f "$pat" 2>/dev/null
    ssh h106 "pkill -9 -f '$pat'" 2>/dev/null
  done
  sleep 5
}

kill_stragglers

ssh h106 "bash -c 'source $P52/scripts/env_2node.sh && export $OVR && W7_NODE_RANK=1 exec $PY $P52/scripts/w7_2node.py spec'" \
    > "$PHASE/logs/prof236b_h106.log" 2>&1 &
H106_PID=$!

source "$P52/scripts/env_2node.sh"
export $OVR
W7_NODE_RANK=0 $PY "$P52/scripts/w7_2node.py" spec \
    > "$PHASE/logs/prof236b_h107.log" 2>&1
RC=$?
wait "$H106_PID"
kill_stragglers
echo "prof236b done RC=$RC"
grep -h 'W7-2N K' "$PHASE/logs/prof236b_h107.log" | tail -1
