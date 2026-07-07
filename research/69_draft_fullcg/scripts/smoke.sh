#!/bin/bash
# Phase 69 fast smoke (DP4/EP4, 2k, W64, shared-KV bf16 self-draft).
# Usage: smoke.sh TAG K FULLCG [EXTRA_ENV]
set -u
TAG="${1:?tag}"; K="${2:-2}"; FULLCG="${3:-1}"; EXTRA="${4:-}"
PHASE=/h/v-sukmincho/self-spec-moe/research/69_draft_fullcg
P52=/h/v-sukmincho/self-spec-moe/research/52_two_node_e2e
PY=/h/v-sukmincho/self-spec-moe/.venv/bin/python
mkdir -p "$PHASE/logs" "$PHASE/data"
for pat in 'w7_2node[.]py' 'EngineCor[e]' 'Worker_D[P]'; do pkill -9 -f "$pat" 2>/dev/null; done
sleep 4
RUN="W7_MODEL=Qwen/Qwen3-30B-A3B W7_NODES=1 W7_LOCAL_WORLD=4 W7_TP=1 \
W7_CTX_TOKENS=2048 W7_MAX_MODEL_LEN=4096 W7_MAX_NUM_BATCHED=4096 \
W7_GPU_MEM=0.85 W7_ITERS=1 W7_WARMUP=1 W7_OUTLEN=64 W7_SHORTLEN=16 \
W7_BATCHES=4 W7_KS=$K W7_DRAFT_QUANT= W7_DRAFT_FULL_REPLICA=0 \
W7_DRAFT_LOCAL_ROUTE=0 W7_DRAFT_NODE_LOCAL=0 \
VLLM_SELF_SPEC_SHARED_KV=1 VLLM_SELF_SPEC_DRAFT_KV_WINDOW=64 \
VLLM_SELF_SPEC_DRAFT_KV_SINKS=16 W7_KV_WINDOW_DEBUG=1 \
VLLM_SELF_SPEC_DRAFT_FULLCG=$FULLCG $EXTRA \
W7_OUT=$PHASE/data W7_TAG=q30b_p69_smoke_$TAG \
W7_MASTER_PORT=15466 W7_NODE_RANK=0"
source "$PHASE/scripts/env_1node.sh"
export $RUN
LOG="$PHASE/logs/smoke_${TAG}.log"
timeout 900 $PY "$P52/scripts/w7_2node.py" spec > "$LOG" 2>&1
echo "=== smoke $TAG (K=$K FULLCG=$FULLCG $EXTRA) exit $? ==="
grep -hE "W7-2N K=|accept_len|scratchpad attention active|Error|Traceback|assert" "$LOG" | tail -8
