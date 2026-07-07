#!/bin/bash
# Phase 69: single-node (h107, DP8/EP8) self-spec run, arm-B config
# (fp8 comm-free replica + shared-KV + W512 + P65 flag stack). Optional
# SelfSpecProfiler + extra env (e.g. VLLM_SELF_SPEC_DRAFT_FULLCG=1) for the
# before/after draft-step-CG comparison.
#
# Usage: run_1node.sh TAG K BATCH FINE [PORT] [EXTRA_ENV]
#   K    -> single K (e.g. 4) or comma list (e.g. 2,4) for the F/D two-point.
#   FINE=off -> NO profiler (clean tok/s). FINE=0 coarse. FINE=1 full regions.
set -u
TAG="${1:?tag}"; K="${2:?K}"; BATCH="${3:?batch}"; FINE="${4:-off}"
PORT="${5:-17000}"; EXTRA="${6:-}"
PHASE=/h/v-sukmincho/self-spec-moe/research/70_fullcg_coverage
P52=/h/v-sukmincho/self-spec-moe/research/52_two_node_e2e
P57=/h/v-sukmincho/self-spec-moe/research/57_large_ep_spec_strategy
PY=/h/v-sukmincho/self-spec-moe/.venv/bin/python
ENVF="$PHASE/scripts/env_1node.sh"
CTX="${W69_CTX:-16384}"; MAXLEN="${W69_MAXLEN:-20480}"; MNB="${W69_MNB:-8192}"
ITERS="${W69_ITERS:-2}"; WARMUP="${W69_WARMUP:-1}"
TRY_TO="${W69_TRYTO:-1800}"; RETRIES="${W69_RETRIES:-3}"
PROMPTS="${W69_PROMPTS:-$P57/data/prompts_ondist.txt}"
PROFDIR="$PHASE/data/prof_${TAG}"
mkdir -p "$PHASE/logs" "$PHASE/data" "$PROFDIR"
rm -f "$PROFDIR"/self_spec_profile_*.json 2>/dev/null

# Arm-B config: fp8 comm-free replica + shared KV + W512 sinks16 + P65 stack.
ARMB="W7_DRAFT_QUANT=fp8 W7_DRAFT_FULL_REPLICA=1 W7_DRAFT_LOCAL_ROUTE=1 \
W7_DRAFT_NODE_LOCAL=0 \
VLLM_SELF_SPEC_SHARED_KV=1 VLLM_SELF_SPEC_DRAFT_KV_WINDOW=512 \
VLLM_SELF_SPEC_DRAFT_KV_SINKS=16 W7_KV_WINDOW_DEBUG=1 \
VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1 \
VLLM_SELF_SPEC_DRAFT_SKIP_DP_COORD=1"

if [ "$FINE" = "off" ]; then PROF="";
else
  PROF="VLLM_SELF_SPEC_PROFILE=1 VLLM_SELF_SPEC_PROFILE_OUT=$PROFDIR \
VLLM_SELF_SPEC_PROFILE_WARMUP=${W69_PWARMUP:-40}"
  [ "$FINE" = "1" ] && PROF="$PROF VLLM_SELF_SPEC_PROFILE_FINE=1"
fi

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
    if [ -n "$u" ] && [ "$u" != "$me" ]; then
      echo "[w69] foreign GPU pid=$p user=$u"; return 0
    fi
  done
  return 1
}

ok=0
for try in $(seq 1 "$RETRIES"); do
  while foreign_gpu_busy; do
    echo "[w69] $TAG: waiting 120s for foreign GPU ($(date +%H:%M:%S))"; sleep 120
  done
  kill_stragglers
  LOG="$PHASE/logs/${TAG}_K${K}_b${BATCH}_try${try}.log"
  OVR="W7_NODES=1 W7_LOCAL_WORLD=8 W7_KS=$K W7_BATCHES=$BATCH \
W7_ITERS=$ITERS W7_WARMUP=$WARMUP W7_GPU_MEM=0.90 W7_TAG=p69_${TAG} \
W7_MASTER_PORT=$((PORT + try * 20)) W7_CTX_TOKENS=$CTX \
W7_MAX_MODEL_LEN=$MAXLEN W7_MAX_NUM_BATCHED=$MNB W7_OUT=$PHASE/data \
W7_PROMPT_FILE=$PROMPTS W7_CHAT=1 $ARMB $PROF $EXTRA"
  echo "[w69] $TAG K=$K b=$BATCH fine=$FINE try=$try to=${TRY_TO}s ($(date +%H:%M:%S))"
  ( source "$ENVF" && export $OVR && \
    W7_NODE_RANK=0 timeout "$TRY_TO" $PY "$P52/scripts/w7_2node.py" spec ) \
      > "$LOG" 2>&1
  nk=$(echo "$K" | awk -F, '{print NF}')
  nb=$(echo "$BATCH" | awk -F, '{print NF}')
  want=$((nk * nb))
  got=$(grep -cE "W7-2N K=.* batch=.* accept_len" "$LOG" 2>/dev/null)
  echo "[w69] $TAG try=$try -> ${got:-0}/$want rows ($(date +%H:%M:%S))"
  grep -hE "W7-2N K=|KV cache size|shared-KV|kv-window|DRAFT_FULLCG|scratchpad" "$LOG" 2>/dev/null | tail -10
  if [ "${got:-0}" -ge "$want" ]; then ok=1; break; fi
done
kill_stragglers
[ "$ok" = "1" ] || { echo "[w69] $TAG INCOMPLETE after $RETRIES tries"; exit 1; }
echo "[w69] $TAG done ($(date +%H:%M:%S))"
