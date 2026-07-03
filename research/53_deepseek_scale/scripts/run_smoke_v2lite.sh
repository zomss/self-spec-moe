#!/bin/bash
# Phase 53 smoke: DeepSeek-V2-Lite, single-node DP2, K=1, FP8 EP-shard
# local-route draft (NO full replica). Validates deepseek_v2 + MLA draft
# chain + local routing in-engine; accept sanity vs Phase 25 ballpark.
set -u
PHASE=/h/v-sukmincho/self-spec-moe/research/53_deepseek_scale
P52=/h/v-sukmincho/self-spec-moe/research/52_two_node_e2e
PY=/h/v-sukmincho/self-spec-moe/.venv/bin/python
mkdir -p "$PHASE/logs" "$PHASE/data"

pkill -9 -f 'w7_2node[.]py' 2>/dev/null; pkill -9 -f 'EngineCor[e]_' 2>/dev/null
sleep 5

source "$P52/scripts/env_2node.sh"
export W7_NODES=1 W7_LOCAL_WORLD=2
export W7_MODEL=deepseek-ai/DeepSeek-V2-Lite W7_TRC=1
export W7_KS=1 W7_BATCHES=8 W7_ITERS=1 W7_WARMUP=1
export W7_DRAFT_FULL_REPLICA=0
export W7_TAG=v2lite_smoke W7_OUT="$PHASE/data"
export W7_MASTER_PORT=13410
MODE="${1:-spec}"
W7_NODE_RANK=0 $PY "$P52/scripts/w7_2node.py" "$MODE" \
    > "$PHASE/logs/smoke_v2lite_${MODE}.log" 2>&1
RC=$?
pkill -9 -f 'EngineCor[e]_' 2>/dev/null
echo "smoke $MODE RC=$RC"
grep -h 'W7-2N' "$PHASE/logs/smoke_v2lite_${MODE}.log" | tail -2
exit $RC
