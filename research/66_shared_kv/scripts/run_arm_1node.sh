#!/bin/bash
# Phase 66: run ONE single-node (h107, DP8/EP8) self-spec arm with the
# shared-KV drafter flag. Clone of research/62_window_kv_draft/scripts/
# run_arm.sh with:
#   W66_SHARED (1)  -> VLLM_SELF_SPEC_SHARED_KV
#   W66_EXTRA ("")  -> extra "K=V ..." env appended (e.g. P65 fix flags)
# Refuses to launch while a FOREIGN GPU compute PID is present (shared node).
# Usage: [W66_* overrides] run_arm_1node.sh ARMTAG PORT
set -u
ARM="${1:?arm tag}"
PORT="${2:?base master port}"
PHASE=/h/v-sukmincho/self-spec-moe/research/66_shared_kv
P52=/h/v-sukmincho/self-spec-moe/research/52_two_node_e2e
P57=/h/v-sukmincho/self-spec-moe/research/57_large_ep_spec_strategy
PY=/h/v-sukmincho/self-spec-moe/.venv/bin/python
ENVF="$PHASE/scripts/env_1node.sh"

K="${W66_K:-4}"; BATCHES="${W66_BATCHES:-8}"
CTX="${W66_CTX:-16384}"; MAXLEN="${W66_MAXLEN:-20480}"
WINDOW="${W66_WINDOW:-0}"; SINKS="${W66_SINKS:-16}"
QUANT="${W66_QUANT:-}"; REPLICA="${W66_REPLICA:-0}"
LOCROUTE="${W66_LOCALROUTE:-0}"; SHARED="${W66_SHARED:-1}"
EXTRA="${W66_EXTRA:-}"
ITERS="${W66_ITERS:-2}"; WARMUP="${W66_WARMUP:-1}"; MNB="${W66_MNB:-8192}"
TRY_TO="${W66_TRYTO:-1800}"; RETRIES="${W66_RETRIES:-3}"
PROMPTS="${W66_PROMPTS:-$P57/data/prompts_ondist.txt}"
mkdir -p "$PHASE/logs" "$PHASE/data"
TAG="w66_${ARM}"

kill_stragglers() {
  # Shared node: pkill OUR bracket-escaped patterns only (own-user procs);
  # never kill arbitrary GPU PIDs, never sudo kill.
  for pat in 'w7_2node[.]py' 'EngineCor[e]' 'Worker_D[P]'; do
    pkill -9 -f "$pat" 2>/dev/null
  done
  sleep 8
}

foreign_gpu_busy() {
  local pids p u me
  me="$(whoami)"
  pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
  for p in $pids; do
    u="$(ps -o user= -p "$p" 2>/dev/null | tr -d ' ')"
    if [ -n "$u" ] && [ "$u" != "$me" ]; then
      echo "[w66] foreign GPU pid=$p user=$u"
      return 0
    fi
  done
  return 1
}

nb=$(echo "$BATCHES" | awk -F, '{print NF}')
ok=0
for try in $(seq 1 "$RETRIES"); do
  while foreign_gpu_busy; do
    echo "[w66] $ARM: waiting 120s for foreign GPU work to finish ($(date +%H:%M:%S))"
    sleep 120
  done
  kill_stragglers
  LOG="$PHASE/logs/${TAG}_K${K}_try${try}.log"
  OVR="W7_NODES=1 W7_LOCAL_WORLD=8 W7_KS=$K W7_BATCHES=$BATCHES \
W7_ITERS=$ITERS W7_WARMUP=$WARMUP W7_GPU_MEM=0.90 W7_TAG=$TAG \
W7_MASTER_PORT=$((PORT + try * 20)) W7_CTX_TOKENS=$CTX \
W7_MAX_MODEL_LEN=$MAXLEN W7_MAX_NUM_BATCHED=$MNB W7_OUT=$PHASE/data \
W7_PROMPT_FILE=$PROMPTS W7_CHAT=1 \
W7_DRAFT_QUANT=$QUANT W7_DRAFT_FULL_REPLICA=$REPLICA \
W7_DRAFT_LOCAL_ROUTE=$LOCROUTE \
VLLM_SELF_SPEC_SHARED_KV=$SHARED \
VLLM_SELF_SPEC_DRAFT_KV_WINDOW=$WINDOW \
VLLM_SELF_SPEC_DRAFT_KV_SINKS=$SINKS W7_KV_WINDOW_DEBUG=1 $EXTRA"
  echo "[w66] $ARM K=$K W=$WINDOW shared=$SHARED quant=${QUANT:-bf16}" \
       "replica=$REPLICA ctx=$CTX try=$try to=${TRY_TO}s ($(date +%H:%M:%S))"
  ( source "$ENVF" && export $OVR && \
    W7_NODE_RANK=0 timeout "$TRY_TO" $PY "$P52/scripts/w7_2node.py" spec ) \
      > "$LOG" 2>&1
  got=$(grep -cE "W7-2N K=$K\] batch=.* accept_len" "$LOG" 2>/dev/null)
  echo "[w66] $ARM try=$try -> ${got:-0}/$nb batch rows ($(date +%H:%M:%S))"
  grep -hE "W7-2N K=$K|KV cache size|shared-KV|kv-window" "$LOG" 2>/dev/null | tail -20
  if [ "${got:-0}" -ge "$nb" ]; then ok=1; break; fi
done
kill_stragglers
[ "$ok" = "1" ] || { echo "[w66] $ARM INCOMPLETE after $RETRIES tries"; exit 1; }
echo "[w66] $ARM done ($(date +%H:%M:%S))"
