#!/bin/bash
# Phase 49 2-node launcher: run on h107; starts the h106 side over SSH.
# Usage: run_2node.sh {nospec|spec}
set -u
MODE="${1:?usage: run_2node.sh nospec|spec}"
PHASE=/h/v-sukmincho/self-spec-moe/research/52_two_node_e2e
PY=/h/v-sukmincho/self-spec-moe/.venv/bin/python
mkdir -p "$PHASE/logs" "$PHASE/data"

kill_stragglers() {
  for host in "" h106; do
    if [ -z "$host" ]; then
      pkill -9 -f "w7_2node" 2>/dev/null
      pkill -9 -f "EngineCore" 2>/dev/null
    else
      ssh "$host" 'pkill -9 -f "w7_2node" 2>/dev/null; pkill -9 -f "EngineCore" 2>/dev/null' 2>/dev/null
    fi
  done
  sleep 5
}

kill_stragglers

ssh h106 "bash -c 'source $PHASE/scripts/env_2node.sh && W7_NODE_RANK=1 exec $PY $PHASE/scripts/w7_2node.py $MODE'" \
    > "$PHASE/logs/${MODE}_h106.log" 2>&1 &
H106_PID=$!

source "$PHASE/scripts/env_2node.sh"
W7_NODE_RANK=0 $PY "$PHASE/scripts/w7_2node.py" "$MODE" \
    > "$PHASE/logs/${MODE}_h107.log" 2>&1
RC=$?
wait "$H106_PID"
kill_stragglers
echo "run_2node $MODE done RC=$RC"
exit $RC
