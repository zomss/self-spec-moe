#!/bin/bash
# Phase 69 accept canary (Stage 3.1): single-node DP8/EP8 2k, W64, K2, shared-KV
# bf16 self-draft. Runs the PIECEWISE reference (scratchpad OFF) and the
# window-scratchpad FULL-CG path (scratchpad ON); greedy accept_len must match
# to ~3 decimals (bit-exact key set). P66 shared-KV ref: W64 K2 = 2.689.
# Usage: run_canary.sh   (runs both arms)
set -u
PHASE=/h/v-sukmincho/self-spec-moe/research/69_draft_fullcg
P52=/h/v-sukmincho/self-spec-moe/research/52_two_node_e2e
P57=/h/v-sukmincho/self-spec-moe/research/57_large_ep_spec_strategy
PY=/h/v-sukmincho/self-spec-moe/.venv/bin/python
ENVF="$PHASE/scripts/env_1node.sh"
mkdir -p "$PHASE/logs" "$PHASE/data"

kill_stragglers() {
  for pat in 'w7_2node[.]py' 'EngineCor[e]' 'Worker_D[P]'; do
    pkill -9 -f "$pat" 2>/dev/null
  done
  sleep 6
}
foreign_gpu_busy() {
  local pids p u me; me="$(whoami)"
  pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
  for p in $pids; do
    u="$(ps -o user= -p "$p" 2>/dev/null | tr -d ' ')"
    [ -n "$u" ] && [ "$u" != "$me" ] && { echo "[w69] foreign GPU pid=$p"; return 0; }
  done
  return 1
}

run_arm() {  # $1=tag $2=FULLCG $3=port
  local TAG="$1" FULLCG="$2" PORT="$3"
  while foreign_gpu_busy; do echo "[w69] wait foreign GPU"; sleep 120; done
  kill_stragglers
  local LOG="$PHASE/logs/canary_${TAG}.log"
  local OVR="W7_NODES=1 W7_LOCAL_WORLD=8 W7_KS=2 W7_BATCHES=8 \
W7_ITERS=2 W7_WARMUP=1 W7_GPU_MEM=0.90 W7_TAG=p69_canary_${TAG} \
W7_MASTER_PORT=$PORT W7_CTX_TOKENS=2048 W7_MAX_MODEL_LEN=2560 \
W7_MAX_NUM_BATCHED=8192 W7_OUT=$PHASE/data W7_PROMPT_FILE=$P57/data/prompts_ondist.txt \
W7_CHAT=1 W7_DRAFT_QUANT= W7_DRAFT_FULL_REPLICA=0 W7_DRAFT_LOCAL_ROUTE=0 \
W7_DRAFT_NODE_LOCAL=0 VLLM_SELF_SPEC_SHARED_KV=1 \
VLLM_SELF_SPEC_DRAFT_KV_WINDOW=64 VLLM_SELF_SPEC_DRAFT_KV_SINKS=16 \
W7_KV_WINDOW_DEBUG=1 VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 \
VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1 VLLM_SELF_SPEC_DRAFT_SKIP_DP_COORD=1 \
VLLM_SELF_SPEC_DRAFT_FULLCG=$FULLCG"
  echo "[w69] canary $TAG FULLCG=$FULLCG ($(date +%H:%M:%S))"
  ( source "$ENVF" && export $OVR && \
    W7_NODE_RANK=0 timeout 1200 $PY "$P52/scripts/w7_2node.py" spec ) > "$LOG" 2>&1
  echo "[w69] canary $TAG ->"
  grep -hE "W7-2N K=|scratchpad|kv-window|accept_len" "$LOG" | tail -6
}

run_arm off 0 16640
run_arm on  1 16660
echo "=== canary A/B done ==="
echo "--- OFF (reference) ---"; grep -hE "W7-2N K=" "$PHASE/logs/canary_off.log" | tail -2
echo "--- ON  (scratchpad) ---"; grep -hE "W7-2N K=" "$PHASE/logs/canary_on.log" | tail -2
