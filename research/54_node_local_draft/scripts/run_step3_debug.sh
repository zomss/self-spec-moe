#!/bin/bash
# Phase 55 debug: step3 b32-only with full per-dispatch a2a logging, to
# locate the rank-divergent collective behind the b32 mixed-step hang.
set -u
PHASE=/h/v-sukmincho/self-spec-moe/research/54_node_local_draft
P52=/h/v-sukmincho/self-spec-moe/research/52_two_node_e2e
PY=/h/v-sukmincho/self-spec-moe/.venv/bin/python
mkdir -p "$PHASE/logs" "$PHASE/data"

OVR="W7_DRAFT_LOCAL_ROUTE=0 W7_DRAFT_NODE_LOCAL=1 W7_DRAFT_FULL_REPLICA=0 \
W7_KS=1 W7_OUT=$PHASE/data VLLM_SELF_SPEC_DRAFT_SKIP_DP_COORD=0 \
VLLM_SELF_SPEC_DRAFT_AMORTIZE_DP_COORD=0 \
VLLM_SELF_SPEC_DRAFT_STEP0_FULL_CG=0 \
W7_MODEL=deepseek-ai/DeepSeek-V2 W7_TRC=1 \
W7_NODES=2 W7_LOCAL_WORLD=4 W7_TP=2 W7_BATCHES=32 W7_ITERS=1 W7_WARMUP=1 \
W7_GPU_MEM=0.90 W7_MAX_MODEL_LEN=1024 W7_CG_SIZES=8,16,32,64,128 W7_MAX_NUM_BATCHED=2048 \
W7_TAG=dbg_dsv2_b32 W7_MASTER_PORT=13435 \
W7_A2A_DEBUG=1 W7_A2A_DEBUG_ALL=${DBGALL:-0}"

kill_stragglers() {
  for pat in 'w7_2node[.]py' 'EngineCor[e]' 'Worker_D[P]' 'VLLM[:]:'; do
    pkill -9 -f "$pat" 2>/dev/null
    ssh h106 "pkill -9 -f '$pat'" 2>/dev/null
  done
  sleep 5
}

kill_stragglers

ssh h106 "bash -c 'source $P52/scripts/env_2node.sh && export $OVR && W7_NODE_RANK=1 exec $PY $P52/scripts/w7_2node.py spec'" \
    > "$PHASE/logs/dbg32_h106.log" 2>&1 &
H106_PID=$!

source "$P52/scripts/env_2node.sh"
export $OVR
W7_NODE_RANK=0 $PY "$P52/scripts/w7_2node.py" spec \
    > "$PHASE/logs/dbg32_h107.log" 2>&1
RC=$?
wait "$H106_PID"
kill_stragglers
echo "dbg32 done RC=$RC"
grep -h 'W7-2N K' "$PHASE/logs/dbg32_h107.log" | tail -2
exit $RC
