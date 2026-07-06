#!/bin/bash
# Phase 59: no-spec Qwen3-30B baseline at a chosen CONTEXT length, 2-node EP16,
# chat-templated on-dist prompts padded to W7_CTX_TOKENS. Apples-to-apples
# denominator for the EAGLE long-context speedup.
# Usage: run_nospec_ctx.sh CTX MAXLEN BATCHES PORT [ITERS] [WARMUP] [MNB]
set -u
CTX="${1:?ctx tokens}"; MAXLEN="${2:?max_model_len}"; BATCHES="${3:-8,32,64}"
PORT="${4:-13700}"; ITERS="${5:-2}"; WARMUP="${6:-1}"; MNB="${7:-4096}"
PHASE=/h/v-sukmincho/self-spec-moe/research/59_multicontext
P57=/h/v-sukmincho/self-spec-moe/research/57_large_ep_spec_strategy
P52=/h/v-sukmincho/self-spec-moe/research/52_two_node_e2e
PY=/h/v-sukmincho/self-spec-moe/.venv/bin/python
ENV="$P57/scripts/env_eagle_2node.sh"
mkdir -p "$PHASE/logs" "$PHASE/data"

kill_stragglers() {
  for pat in 'w7_2node[.]py' 'EngineCor[e]' 'Worker_D[P]' 'VLLM[:]:'; do
    pkill -9 -f "$pat" 2>/dev/null; ssh h106 "pkill -9 -f '$pat'" 2>/dev/null
  done
  sleep 6
}

CKB="$((CTX/1024))k"
TAG="q30b_nospec_2n_ctx${CKB}"
OVR="W7_NODES=2 W7_LOCAL_WORLD=8 W7_BATCHES=$BATCHES W7_ITERS=$ITERS W7_WARMUP=$WARMUP \
W7_GPU_MEM=0.90 W7_TAG=$TAG W7_MASTER_PORT=$PORT W7_CTX_TOKENS=$CTX \
W7_MAX_MODEL_LEN=$MAXLEN W7_MAX_NUM_BATCHED=$MNB W7_OUT=$PHASE/data \
W7_PROMPT_FILE=$P57/data/prompts_ondist.txt W7_CHAT=1"

kill_stragglers
echo "[nospec-ctx] CTX=$CTX MAXLEN=$MAXLEN BATCHES=$BATCHES iters=$ITERS warmup=$WARMUP ($(date +%H:%M:%S))"
ssh h106 "bash -c 'source $ENV && export $OVR && W7_NODE_RANK=1 exec $PY $P52/scripts/w7_2node.py nospec'" \
    > "$PHASE/logs/${TAG}_h106.log" 2>&1 &
H6=$!
( source "$ENV" && export $OVR && W7_NODE_RANK=0 $PY "$P52/scripts/w7_2node.py" nospec ) \
    > "$PHASE/logs/${TAG}_h107.log" 2>&1
RC=$?
wait $H6 2>/dev/null
kill_stragglers
echo "[nospec-ctx] CTX=$CTX done RC=$RC ($(date +%H:%M:%S))"
grep -hE 'W7-2N|ctx]' "$PHASE/logs/${TAG}_h107.log" | tail -8
