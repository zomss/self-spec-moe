#!/bin/bash
# Phase 49 trace driver: spec b64 + nospec b192 torch-profiler traces, 1 node.
set -u
PHASE=/h/v-sukmincho/self-spec-moe/research/52_two_node_e2e
PY=/h/v-sukmincho/self-spec-moe/.venv/bin/python
mkdir -p "$PHASE/logs"

clean() {
  pkill -9 -f 'w7_trace[.]py' 2>/dev/null
  pkill -9 -f 'EngineCor[e]_' 2>/dev/null
  sleep 5
}

source "$PHASE/scripts/env_2node.sh"
export W7_NODES=1 W7_LOCAL_WORLD=8

clean
W7_BATCH=64 W7_TRACE_DIR="$PHASE/data/trace_spec" \
  $PY "$PHASE/scripts/w7_trace.py" spec > "$PHASE/logs/trace_spec.log" 2>&1
RC1=$?
clean
W7_BATCH=192 W7_TRACE_DIR="$PHASE/data/trace_nospec" \
  $PY "$PHASE/scripts/w7_trace.py" nospec > "$PHASE/logs/trace_nospec.log" 2>&1
RC2=$?
clean
echo "run_trace done spec=$RC1 nospec=$RC2"
