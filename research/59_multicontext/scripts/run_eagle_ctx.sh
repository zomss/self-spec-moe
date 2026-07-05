#!/bin/bash
# Phase 59: EAGLE3 K-sweep at a chosen CONTEXT length, 2-node EP16, chat + on-dist
# prompts padded to W7_CTX_TOKENS. ONE K per engine invocation, hard `timeout`,
# GPU-pid force-kill + retry (the DP16-EAGLE first-collective wedge mitigation
# from Phase 57 run_ksweep.sh). Per-K JSON persists under phase data/.
# Usage: run_eagle_ctx.sh CTX MAXLEN BATCHES "KLIST" PORT [ITERS] [WARMUP] [MNB] [TRY_TO] [RETRIES]
set -u
CTX="${1:?ctx tokens}"; MAXLEN="${2:?max_model_len}"; BATCHES="${3:-8,32,64}"
KLIST="${4:-1 2}"; PORT="${5:-13760}"; ITERS="${6:-2}"; WARMUP="${7:-1}"
MNB="${8:-4096}"; TRY_TO="${9:-1200}"; RETRIES="${10:-4}"
PHASE=/h/v-sukmincho/self-spec-moe/research/59_multicontext
P57=/h/v-sukmincho/self-spec-moe/research/57_large_ep_spec_strategy
P52=/h/v-sukmincho/self-spec-moe/research/52_two_node_e2e
PY=/h/v-sukmincho/self-spec-moe/.venv/bin/python
ENV="$P57/scripts/env_eagle_2node.sh"
mkdir -p "$PHASE/logs" "$PHASE/data"

CKB="$((CTX/1024))k"
TAG="q30b_eagle_2n_ctx${CKB}"

kill_stragglers() {
  for pat in 'w7_2node[.]py' 'EngineCor[e]' 'Worker_D[P]' 'VLLM[:]:'; do
    pkill -9 -f "$pat" 2>/dev/null; ssh h106 "pkill -9 -f '$pat'" 2>/dev/null
  done
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u); do kill -9 "$p" 2>/dev/null; done
  ssh h106 'for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u); do kill -9 "$p" 2>/dev/null; done' 2>/dev/null
  sleep 8
}

nb=$(echo "$BATCHES" | awk -F, '{print NF}')
for K in $KLIST; do
  ok=0
  for try in $(seq 1 "$RETRIES"); do
    kill_stragglers
    LOGH="$PHASE/logs/${TAG}_K${K}_try${try}.log"
    LOG6="$PHASE/logs/${TAG}_K${K}_try${try}_h106.log"
    OVR="W7_NODES=2 W7_LOCAL_WORLD=8 W7_KS=$K W7_BATCHES=$BATCHES \
W7_ITERS=$ITERS W7_WARMUP=$WARMUP W7_GPU_MEM=0.90 W7_TAG=$TAG \
W7_MASTER_PORT=$((PORT+try*20)) W7_CTX_TOKENS=$CTX W7_MAX_MODEL_LEN=$MAXLEN \
W7_MAX_NUM_BATCHED=$MNB W7_PROMPT_FILE=$P57/data/prompts_ondist.txt W7_CHAT=1"
    echo "[eagle-ctx] $TAG K=$K try=$try to=${TRY_TO}s ($(date +%H:%M:%S))"
    ssh h106 "bash -c 'source $ENV && export $OVR && W7_NODE_RANK=1 timeout $TRY_TO $PY $P52/scripts/w7_2node.py spec'" \
        > "$LOG6" 2>&1 &
    H6=$!
    ( source "$ENV" && export $OVR && W7_NODE_RANK=0 timeout "$TRY_TO" $PY "$P52/scripts/w7_2node.py" spec ) \
        > "$LOGH" 2>&1
    wait "$H6" 2>/dev/null
    got=$(grep -cE "W7-2N K=$K\] batch=.* accept_len" "$LOGH" 2>/dev/null)
    echo "[eagle-ctx] K=$K try=$try -> ${got:-0}/$nb batch rows ($(date +%H:%M:%S))"
    grep -hE "W7-2N K=$K|ctx\]" "$LOGH" 2>/dev/null
    if [ "${got:-0}" -ge "$nb" ]; then ok=1; break; fi
  done
  [ "$ok" = "1" ] || echo "[eagle-ctx] K=$K INCOMPLETE after $RETRIES tries"
done
kill_stragglers
echo "[eagle-ctx] $TAG done ($(date +%H:%M:%S))"
