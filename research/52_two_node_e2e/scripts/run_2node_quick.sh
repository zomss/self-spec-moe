#!/bin/bash
# Phase 52 quick 2-node driver: b64 only, fresh tag.
# Usage: [QB=..] [QTAG=..] [SKIPDP=0|1] run_2node_quick.sh {nospec|spec}
set -u
MODE="${1:?usage: run_2node_quick.sh nospec|spec}"
PHASE=/h/v-sukmincho/self-spec-moe/research/52_two_node_e2e
PY=/h/v-sukmincho/self-spec-moe/.venv/bin/python
mkdir -p "$PHASE/logs" "$PHASE/data"

OVR="W7_BATCHES=${QB:-64} W7_TAG=${QTAG:-qwen30b_2node_perf} VLLM_SELF_SPEC_DRAFT_SKIP_DP_COORD=${SKIPDP:-0}"

kill_stragglers() {
  pkill -9 -f 'w7_2node[.]py' 2>/dev/null
  pkill -9 -f 'EngineCor[e]_' 2>/dev/null
  ssh h106 "pkill -9 -f 'w7_2node[.]py'; pkill -9 -f 'EngineCor[e]_'" 2>/dev/null
  sleep 5
}

kill_stragglers

ssh h106 "bash -c 'source $PHASE/scripts/env_2node.sh && export $OVR && W7_NODE_RANK=1 exec $PY $PHASE/scripts/w7_2node.py $MODE'" \
    > "$PHASE/logs/perf_${MODE}_h106.log" 2>&1 &
H106_PID=$!

source "$PHASE/scripts/env_2node.sh"
export $OVR
W7_NODE_RANK=0 $PY "$PHASE/scripts/w7_2node.py" "$MODE" \
    > "$PHASE/logs/perf_${MODE}_h107.log" 2>&1
RC=$?
wait "$H106_PID"
kill_stragglers
echo "run_2node_quick $MODE done RC=$RC"
exit $RC
