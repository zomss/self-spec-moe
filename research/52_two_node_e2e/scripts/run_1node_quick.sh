#!/bin/bash
# Phase 52: quick single-node DP8 spec run (A/B for DRAFT_SKIP_DP_COORD).
# Usage: [SKIPDP=1] [QTAG=...] run_1node_quick.sh {nospec|spec}
set -u
MODE="${1:?usage: run_1node_quick.sh nospec|spec}"
PHASE=/h/v-sukmincho/self-spec-moe/research/52_two_node_e2e
PY=/h/v-sukmincho/self-spec-moe/.venv/bin/python
mkdir -p "$PHASE/logs" "$PHASE/data"

pkill -9 -f 'w7_2node[.]py' 2>/dev/null; pkill -9 -f 'EngineCor[e]_' 2>/dev/null
sleep 5

source "$PHASE/scripts/env_2node.sh"
export W7_NODES=1 W7_LOCAL_WORLD=8 W7_BATCHES=64
export W7_TAG="${QTAG:-qwen30b_1node_quick}"
export VLLM_SELF_SPEC_DRAFT_SKIP_DP_COORD="${SKIPDP:-0}"
W7_NODE_RANK=0 $PY "$PHASE/scripts/w7_2node.py" "$MODE" \
    > "$PHASE/logs/quick1n_${MODE}_${W7_TAG}.log" 2>&1
RC=$?
pkill -9 -f 'EngineCor[e]_' 2>/dev/null
echo "run_1node_quick $MODE tag=$W7_TAG skipdp=${SKIPDP:-0} RC=$RC"
exit $RC
