#!/bin/bash
# Phase 54 test ladder for the node-local draft.
# Usage: run_ladder.sh {step1|step2|step3}
#   step1: 1-node V2-Lite DP8, node-local (invariant: node==all -> full-route
#          accept, expect ~1.9 at K=1; validates group + dispatch end-to-end)
#   step2: 2-node V2-Lite DP16/EP16, node-local (50% coverage, expect ~1.88)
#   step3: 2-node DeepSeek-V2 236B TP2xDP8/EP16, node-local (the win test)
set -u
STEP="${1:?usage: run_ladder.sh step1|step2|step3}"
PHASE=/h/v-sukmincho/self-spec-moe/research/54_node_local_draft
P52=/h/v-sukmincho/self-spec-moe/research/52_two_node_e2e
PY=/h/v-sukmincho/self-spec-moe/.venv/bin/python
mkdir -p "$PHASE/logs" "$PHASE/data"

COMMON="W7_DRAFT_LOCAL_ROUTE=0 W7_DRAFT_NODE_LOCAL=1 W7_DRAFT_FULL_REPLICA=0 \
W7_KS=1 W7_OUT=$PHASE/data VLLM_SELF_SPEC_DRAFT_SKIP_DP_COORD=${SKIPDP:-0} \
VLLM_SELF_SPEC_DRAFT_AMORTIZE_DP_COORD=${AMORT:-0}"

case "$STEP" in
  step1)
    OVR="$COMMON W7_MODEL=deepseek-ai/DeepSeek-V2-Lite W7_TRC=1 \
W7_NODES=1 W7_LOCAL_WORLD=8 W7_TP=1 W7_BATCHES=8 W7_ITERS=1 W7_WARMUP=1 \
W7_TAG=ladder1_v2lite_1n W7_MASTER_PORT=13430"
    TWO_NODE=0 ;;
  step2)
    OVR="$COMMON W7_MODEL=deepseek-ai/DeepSeek-V2-Lite W7_TRC=1 \
W7_NODES=2 W7_LOCAL_WORLD=8 W7_TP=1 W7_BATCHES=8 W7_ITERS=1 W7_WARMUP=1 \
W7_TAG=ladder2_v2lite_2n W7_MASTER_PORT=13431"
    TWO_NODE=1 ;;
  step3)
    OVR="$COMMON W7_MODEL=deepseek-ai/DeepSeek-V2 W7_TRC=1 \
W7_NODES=2 W7_LOCAL_WORLD=4 W7_TP=2 W7_BATCHES=8,32,64 W7_ITERS=2 W7_WARMUP=1 \
W7_GPU_MEM=0.90 W7_MAX_MODEL_LEN=1024 W7_CG_SIZES=8,16,32,64,128 W7_MAX_NUM_BATCHED=2048 \
W7_TAG=ladder3_dsv2_2n W7_MASTER_PORT=13432"
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
  ssh h106 "bash -c 'source $P52/scripts/env_2node.sh && export $OVR && W7_NODE_RANK=1 exec $PY $P52/scripts/w7_2node.py spec'" \
      > "$PHASE/logs/${STEP}_h106.log" 2>&1 &
  H106_PID=$!
fi

source "$P52/scripts/env_2node.sh"
export $OVR
W7_NODE_RANK=0 $PY "$P52/scripts/w7_2node.py" spec \
    > "$PHASE/logs/${STEP}_h107.log" 2>&1
RC=$?
[ "$TWO_NODE" = "1" ] && wait "$H106_PID"
kill_stragglers
echo "ladder $STEP done RC=$RC"
grep -h 'W7-2N K' "$PHASE/logs/${STEP}_h107.log" | tail -3
exit $RC
