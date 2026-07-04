#!/bin/bash
# Phase 55: K=2 drain-wedge repro for the node-local draft under SP/TP2.
# V2-Lite, 2 nodes x 8 GPUs, TP2 x DP4-per-node -> DP8/EP16 (the exact 236B
# step3 group topology) with W7_KS=2: the draft chain replays its FULL graph
# (node collectives in-graph) on busy ranks while drained ranks run idle
# dummies -- unfixed, the run wedges at the first mixed drain step; fixed, it
# completes. DBG=1 adds per-coordination + graph-replay + a2a sequence logs.
set -u
PHASE=/h/v-sukmincho/self-spec-moe/research/54_node_local_draft
P52=/h/v-sukmincho/self-spec-moe/research/52_two_node_e2e
PY=/h/v-sukmincho/self-spec-moe/.venv/bin/python
mkdir -p "$PHASE/logs" "$PHASE/data"
TAG="${TAG:-tp2_k2}"

DBGENV=""
[ "${DBG:-0}" = "1" ] && DBGENV="W7_STEP0_DEBUG=1 W7_A2A_DEBUG=1"

OVR="W7_DRAFT_LOCAL_ROUTE=0 W7_DRAFT_NODE_LOCAL=1 W7_DRAFT_FULL_REPLICA=0 \
W7_KS=2 W7_OUT=$PHASE/data VLLM_SELF_SPEC_DRAFT_SKIP_DP_COORD=0 \
VLLM_SELF_SPEC_DRAFT_AMORTIZE_DP_COORD=0 \
VLLM_SELF_SPEC_DRAFT_STEP0_FULL_CG=0 \
W7_MODEL=deepseek-ai/DeepSeek-V2-Lite W7_TRC=1 \
W7_NODES=2 W7_LOCAL_WORLD=4 W7_TP=2 W7_BATCHES=${BATCHES:-8} W7_ITERS=1 W7_WARMUP=1 \
${CGS:+W7_CG_SIZES=$CGS} \
W7_TAG=${TAG}_v2lite_2n W7_MASTER_PORT=13436 $DBGENV"

kill_stragglers() {
  for pat in 'w7_2node[.]py' 'EngineCor[e]' 'Worker_D[P]' 'VLLM[:]:'; do
    pkill -9 -f "$pat" 2>/dev/null
    ssh h106 "pkill -9 -f '$pat'" 2>/dev/null
  done
  sleep 5
}

kill_stragglers

ssh h106 "bash -c 'source $P52/scripts/env_2node.sh && export $OVR && W7_NODE_RANK=1 exec $PY $P52/scripts/w7_2node.py spec'" \
    > "$PHASE/logs/${TAG}_h106.log" 2>&1 &
H106_PID=$!

source "$P52/scripts/env_2node.sh"
export $OVR
W7_NODE_RANK=0 $PY "$P52/scripts/w7_2node.py" spec \
    > "$PHASE/logs/${TAG}_h107.log" 2>&1
RC=$?
wait "$H106_PID"
kill_stragglers
echo "tp2-k2 $TAG done RC=$RC"
grep -h 'W7-2N K' "$PHASE/logs/${TAG}_h107.log" | tail -3
exit $RC
