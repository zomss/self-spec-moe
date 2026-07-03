#!/bin/bash
# Phase 53: DeepSeek-V2 (236B) 2-node run. TP2 x DP4/node -> DP8, EP16.
# Usage: run_2node_dsv2.sh {nospec|spec}
set -u
MODE="${1:?usage: run_2node_dsv2.sh nospec|spec}"
PHASE=/h/v-sukmincho/self-spec-moe/research/53_deepseek_scale
P52=/h/v-sukmincho/self-spec-moe/research/52_two_node_e2e
PY=/h/v-sukmincho/self-spec-moe/.venv/bin/python
mkdir -p "$PHASE/logs" "$PHASE/data"

OVR="W7_MODEL=deepseek-ai/DeepSeek-V2 W7_TRC=1 W7_TP=2 \
W7_NODES=2 W7_LOCAL_WORLD=4 W7_KS=1 W7_BATCHES=8,32,64 \
W7_ITERS=2 W7_WARMUP=1 W7_GPU_MEM=0.90 W7_MAX_MODEL_LEN=1024 W7_CG_SIZES=8,16,32,64,128 W7_MAX_NUM_BATCHED=2048 W7_DRAFT_FULL_REPLICA=0 \
W7_TAG=dsv2_2node W7_OUT=$PHASE/data W7_MASTER_PORT=13420"

kill_stragglers() {
  for pat in 'w7_2node[.]py' 'EngineCor[e]' 'Worker_D[P]' 'VLLM[:]:'; do
    pkill -9 -f "$pat" 2>/dev/null
    ssh h106 "pkill -9 -f '$pat'" 2>/dev/null
  done
  sleep 5
}

kill_stragglers

ssh h106 "bash -c 'source $P52/scripts/env_2node.sh && export $OVR && W7_NODE_RANK=1 exec $PY $P52/scripts/w7_2node.py $MODE'" \
    > "$PHASE/logs/dsv2_${MODE}_h106.log" 2>&1 &
H106_PID=$!

source "$P52/scripts/env_2node.sh"
export $OVR
W7_NODE_RANK=0 $PY "$P52/scripts/w7_2node.py" "$MODE" \
    > "$PHASE/logs/dsv2_${MODE}_h107.log" 2>&1
RC=$?
wait "$H106_PID"
kill_stragglers
echo "run_2node_dsv2 $MODE done RC=$RC"
exit $RC
