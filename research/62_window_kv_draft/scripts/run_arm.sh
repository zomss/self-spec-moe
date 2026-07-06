#!/bin/bash
# Phase 62: run ONE self-spec window-KV arm on single-node DP8/EP8 (h107).
# Hard `timeout` + own-pattern straggler kill + retry with bumped master port
# (run_ksweep.sh pattern, 1-node mode). Refuses to launch while a FOREIGN
# GPU compute PID is present (shared node).
# Usage: [W62_* overrides] run_arm.sh ARMTAG PORT
#   W62_K (4) W62_BATCHES (8) W62_CTX (16384) W62_MAXLEN (20480)
#   W62_WINDOW (0) W62_SINKS (16) W62_QUANT ("") W62_REPLICA (0)
#   W62_LOCALROUTE (0) W62_ITERS (2) W62_WARMUP (1) W62_MNB (8192)
#   W62_TRYTO (1800) W62_RETRIES (3) W62_PROMPTS (phase-57 ondist)
set -u
ARM="${1:?arm tag}"
PORT="${2:?base master port}"
PHASE=/h/v-sukmincho/self-spec-moe/research/62_window_kv_draft
P52=/h/v-sukmincho/self-spec-moe/research/52_two_node_e2e
P57=/h/v-sukmincho/self-spec-moe/research/57_large_ep_spec_strategy
PY=/h/v-sukmincho/self-spec-moe/.venv/bin/python
ENVF="$PHASE/scripts/env_1node.sh"

K="${W62_K:-4}"; BATCHES="${W62_BATCHES:-8}"
CTX="${W62_CTX:-16384}"; MAXLEN="${W62_MAXLEN:-20480}"
WINDOW="${W62_WINDOW:-0}"; SINKS="${W62_SINKS:-16}"
QUANT="${W62_QUANT:-}"; REPLICA="${W62_REPLICA:-0}"
LOCROUTE="${W62_LOCALROUTE:-0}"
ITERS="${W62_ITERS:-2}"; WARMUP="${W62_WARMUP:-1}"; MNB="${W62_MNB:-8192}"
TRY_TO="${W62_TRYTO:-1800}"; RETRIES="${W62_RETRIES:-3}"
PROMPTS="${W62_PROMPTS:-$P57/data/prompts_ondist.txt}"
mkdir -p "$PHASE/logs" "$PHASE/data"
TAG="w62_${ARM}"

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
      echo "[w62] foreign GPU pid=$p user=$u"
      return 0
    fi
  done
  return 1
}

nb=$(echo "$BATCHES" | awk -F, '{print NF}')
ok=0
for try in $(seq 1 "$RETRIES"); do
  while foreign_gpu_busy; do
    echo "[w62] $ARM: waiting 120s for foreign GPU work to finish ($(date +%H:%M:%S))"
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
VLLM_SELF_SPEC_DRAFT_KV_WINDOW=$WINDOW \
VLLM_SELF_SPEC_DRAFT_KV_SINKS=$SINKS W7_KV_WINDOW_DEBUG=1"
  echo "[w62] $ARM K=$K W=$WINDOW quant=${QUANT:-bf16} replica=$REPLICA" \
       "ctx=$CTX try=$try to=${TRY_TO}s ($(date +%H:%M:%S))"
  ( source "$ENVF" && export $OVR && \
    W7_NODE_RANK=0 timeout "$TRY_TO" $PY "$P52/scripts/w7_2node.py" spec ) \
      > "$LOG" 2>&1
  got=$(grep -cE "W7-2N K=$K\] batch=.* accept_len" "$LOG" 2>/dev/null)
  echo "[w62] $ARM try=$try -> ${got:-0}/$nb batch rows ($(date +%H:%M:%S))"
  grep -hE "W7-2N K=$K|ctx\]|kv-window" "$LOG" 2>/dev/null | tail -20
  if [ "${got:-0}" -ge "$nb" ]; then ok=1; break; fi
done
kill_stragglers
[ "$ok" = "1" ] || { echo "[w62] $ARM INCOMPLETE after $RETRIES tries"; exit 1; }
echo "[w62] $ARM done ($(date +%H:%M:%S))"
