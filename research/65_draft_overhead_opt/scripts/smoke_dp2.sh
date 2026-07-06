#!/bin/bash
# Phase 65 smoke: single-node DP2/EP2 self-spec (Qwen3-30B-A3B, draft_model,
# W=512) to validate a stage's flags end-to-end (engine up, propose runs,
# accept sane) before a 2-node 16k launch. ~2k ctx so it is fast.
# (Dense models refuse offline DP, so the smoke uses the real MoE at DP2.)
# Usage: smoke_dp2.sh {base|f1|f12|f123} [K=2]
set -u
STAGE="${1:?base|f1|f12|f123}"
K="${2:-2}"
PHASE=/h/v-sukmincho/self-spec-moe/research/65_draft_overhead_opt
P52=/h/v-sukmincho/self-spec-moe/research/52_two_node_e2e
PY=/h/v-sukmincho/self-spec-moe/.venv/bin/python
case "$STAGE" in
  base) FIX="" ;;
  f1)   FIX="VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1" ;;
  f12)  FIX="VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 \
VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1" ;;
  f123) FIX="VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 \
VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1" ;;
  *) echo "unknown stage $STAGE"; exit 2 ;;
esac
QUANT="W7_DRAFT_QUANT= W7_DRAFT_FULL_REPLICA=0 W7_DRAFT_LOCAL_ROUTE=0"
[ "$STAGE" = "f123" ] && QUANT="W7_DRAFT_QUANT=fp8 W7_DRAFT_FULL_REPLICA=1 \
W7_DRAFT_LOCAL_ROUTE=1"
RUN="W7_MODEL=Qwen/Qwen3-30B-A3B W7_NODES=1 W7_LOCAL_WORLD=4 W7_TP=1 \
W7_CTX_TOKENS=2048 W7_MAX_MODEL_LEN=4096 W7_MAX_NUM_BATCHED=4096 \
W7_GPU_MEM=0.85 W7_ITERS=1 W7_WARMUP=1 W7_OUTLEN=64 W7_SHORTLEN=16 \
W7_BATCHES=4 W7_KS=$K $QUANT $FIX W7_DRAFT_NODE_LOCAL=0 \
VLLM_SELF_SPEC_DRAFT_KV_WINDOW=512 VLLM_SELF_SPEC_DRAFT_KV_SINKS=16 \
W7_KV_WINDOW_DEBUG=1 W7_OUT=$PHASE/data W7_TAG=q30b_p65_smoke_$STAGE \
W7_MASTER_PORT=15444 W7_NODE_RANK=0"
source "$PHASE/scripts/env_selfspec_2node.sh"
export $RUN
exec timeout 900 $PY "$P52/scripts/w7_2node.py" spec
