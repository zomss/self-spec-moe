#!/bin/bash
# Phase 56 Stage A physics at 236B: DeepSeek-V2 2-node TP2xDP4/node EP16,
# K=2 (the ahead chain requires K>=2), gpu_mem 0.95, node-local draft.
# AHEAD=0|1 selects the arm; AHEAD_STEPS optionally caps the ahead work
# (W7_AHEAD_STEPS=1 -> ~1 draft forward: the clean overlap discriminator).
# PROFILE=1 adds the self-spec profiler (region syncs SERIALIZE the overlap:
# profiled runs are component attribution only, never the physics A/B).
set -u
PHASE=/h/v-sukmincho/self-spec-moe/research/56_overlap_node_local
P52=/h/v-sukmincho/self-spec-moe/research/52_two_node_e2e
PY=/h/v-sukmincho/self-spec-moe/.venv/bin/python
mkdir -p "$PHASE/logs" "$PHASE/data"
AHEAD="${AHEAD:-1}"
PROFILE="${PROFILE:-0}"
TAG="${TAG:-236b_a${AHEAD}${AHEAD_STEPS:+_s$AHEAD_STEPS}$( [ "$PROFILE" = 1 ] && echo _prof )}"
PROFDIR="$PHASE/data/prof_${TAG}"
[ "$PROFILE" = 1 ] && { rm -rf "$PROFDIR"; mkdir -p "$PROFDIR"; ssh h106 "rm -rf $PROFDIR && mkdir -p $PROFDIR"; }

OVR="W7_DRAFT_LOCAL_ROUTE=0 W7_DRAFT_NODE_LOCAL=1 W7_DRAFT_FULL_REPLICA=0 \
W7_KS=2 W7_OUT=$PHASE/data VLLM_SELF_SPEC_DRAFT_SKIP_DP_COORD=0 \
VLLM_SELF_SPEC_DRAFT_AMORTIZE_DP_COORD=0 \
VLLM_SELF_SPEC_DRAFT_STEP0_FULL_CG=0 \
VLLM_SELF_SPEC_DRAFT_GRAPH_POOL=1 VLLM_SELF_SPEC_DRAFT_WORKSPACE=1 \
VLLM_SELF_SPEC_AHEAD_CHAIN=$AHEAD \
${AHEAD_STEPS:+W7_AHEAD_STEPS=$AHEAD_STEPS} \
VLLM_SELF_SPEC_PROFILE=$PROFILE \
$( [ "$PROFILE" = 1 ] && echo VLLM_SELF_SPEC_PROFILE_OUT=$PROFDIR ) \
W7_MODEL=deepseek-ai/DeepSeek-V2 W7_TRC=1 \
W7_NODES=2 W7_LOCAL_WORLD=4 W7_TP=2 W7_BATCHES=${BATCHES:-8,64} \
W7_ITERS=${ITERS:-2} W7_WARMUP=1 \
W7_GPU_MEM=0.95 W7_MAX_MODEL_LEN=1024 W7_CG_SIZES=8,16,32,64,128 W7_MAX_NUM_BATCHED=2048 \
W7_TAG=${TAG} W7_MASTER_PORT=1346$AHEAD"

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
W7_NODE_RANK=0 timeout "${RUN_TIMEOUT:-3600}" $PY "$P52/scripts/w7_2node.py" spec \
    > "$PHASE/logs/${TAG}_h107.log" 2>&1
RC=$?
if [ "$RC" = "124" ]; then
  PYSPY=/h/v-sukmincho/self-spec-moe/.venv/bin/py-spy
  for pid in $(pgrep -f 'w7_2node[.]py'); do
    $PYSPY dump --pid "$pid" >> "$PHASE/logs/${TAG}_wedge_h107.txt" 2>&1
  done
  ssh h106 "for pid in \$(pgrep -f 'w7_2node[.]py'); do $PYSPY dump --pid \$pid; done" \
      >> "$PHASE/logs/${TAG}_wedge_h106.txt" 2>&1
fi
kill_stragglers
wait "$H106_PID" 2>/dev/null
echo "236b AHEAD=$AHEAD PROFILE=$PROFILE done RC=$RC"
grep -h 'W7-2N K' "$PHASE/logs/${TAG}_h107.log" | tail -4
grep -h 'ahead-chain hit rate' "$PHASE/logs/${TAG}_h107.log" | tail -3
exit $RC
