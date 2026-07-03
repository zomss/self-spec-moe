#!/bin/bash
# Phase 49: single-node DP8 spec profile (native NVLink) — the comparison
# point for the 2-node cycle decomposition. Same stack, same overrides.
set -u
PHASE=/h/v-sukmincho/self-spec-moe/research/52_two_node_e2e
PY=/h/v-sukmincho/self-spec-moe/.venv/bin/python
mkdir -p "$PHASE/logs" "$PHASE/data/profile1n"

pkill -9 -f "w7_2node" 2>/dev/null; pkill -9 -f "EngineCore" 2>/dev/null
sleep 5

source "$PHASE/scripts/env_2node.sh"
export W7_BATCHES=64 W7_ITERS=1 W7_WARMUP=1 W7_TAG=qwen30b_1node_prof
export W7_NODES=1 W7_LOCAL_WORLD=8
export VLLM_SELF_SPEC_PROFILE=1
export VLLM_SELF_SPEC_PROFILE_OUT="$PHASE/data/profile1n"
W7_NODE_RANK=0 $PY "$PHASE/scripts/w7_2node.py" spec \
    > "$PHASE/logs/prof_1node.log" 2>&1
RC=$?
pkill -9 -f "EngineCore" 2>/dev/null
echo "run_1node_profile done RC=$RC"
exit $RC
