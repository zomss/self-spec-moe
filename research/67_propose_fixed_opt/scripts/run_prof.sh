#!/bin/bash
# Phase 67: single-node (h107, DP8/EP8) self-spec run with the arm-B config
# (fp8 comm-free replica + shared-KV + W512 + P65 flag stack) and the
# SelfSpecProfiler enabled so the per-cycle propose-fixed block decomposes
# into named regions. Writes the harness JSON (data/) + per-PID profiler
# JSONs (data/prof_<TAG>/) so analyze_prof.py can print the F-decomposition.
#
# Usage: run_prof.sh TAG K BATCH MODE [PORT] [EXTRA_ENV]
#   K    -> single K (e.g. 4) or comma list (e.g. 2,4) for the F/D two-point.
#   MODE=off   -> NO profiler (clean tok/s for the F/D two-point solve).
#   MODE=0     -> coarse regions only (verify/draft_chain/draft_forward).
#   MODE=1     -> full per-region decomposition incl. cpu_* glue (attribution
#                only; extra syncs, not for headline tok/s). Use single K.
set -u
TAG="${1:?tag}"; K="${2:?K}"; BATCH="${3:?batch}"; FINE="${4:-0}"
PORT="${5:-16800}"; EXTRA="${6:-}"
PHASE=/h/v-sukmincho/self-spec-moe/research/67_propose_fixed_opt
P52=/h/v-sukmincho/self-spec-moe/research/52_two_node_e2e
P57=/h/v-sukmincho/self-spec-moe/research/57_large_ep_spec_strategy
PY=/h/v-sukmincho/self-spec-moe/.venv/bin/python
ENVF="$PHASE/scripts/env_1node.sh"
CTX="${W67_CTX:-16384}"; MAXLEN="${W67_MAXLEN:-20480}"; MNB="${W67_MNB:-8192}"
ITERS="${W67_ITERS:-2}"; WARMUP="${W67_WARMUP:-1}"
TRY_TO="${W67_TRYTO:-1800}"; RETRIES="${W67_RETRIES:-3}"
PROMPTS="${W67_PROMPTS:-$P57/data/prompts_ondist.txt}"
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

if [ "$FINE" = "off" ]; then
  PROF=""
else
  PROF="VLLM_SELF_SPEC_PROFILE=1 VLLM_SELF_SPEC_PROFILE_OUT=$PROFDIR \
VLLM_SELF_SPEC_PROFILE_WARMUP=${W67_PWARMUP:-40}"
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
      echo "[w67] foreign GPU pid=$p user=$u"; return 0
    fi
  done
  return 1
}

ok=0
for try in $(seq 1 "$RETRIES"); do
  while foreign_gpu_busy; do
    echo "[w67] $TAG: waiting 120s for foreign GPU ($(date +%H:%M:%S))"; sleep 120
  done
  kill_stragglers
  LOG="$PHASE/logs/${TAG}_K${K}_b${BATCH}_try${try}.log"
  OVR="W7_NODES=1 W7_LOCAL_WORLD=8 W7_KS=$K W7_BATCHES=$BATCH \
W7_ITERS=$ITERS W7_WARMUP=$WARMUP W7_GPU_MEM=0.90 W7_TAG=p67_${TAG} \
W7_MASTER_PORT=$((PORT + try * 20)) W7_CTX_TOKENS=$CTX \
W7_MAX_MODEL_LEN=$MAXLEN W7_MAX_NUM_BATCHED=$MNB W7_OUT=$PHASE/data \
W7_PROMPT_FILE=$PROMPTS W7_CHAT=1 $ARMB $PROF $EXTRA"
  echo "[w67] $TAG K=$K b=$BATCH fine=$FINE try=$try to=${TRY_TO}s ($(date +%H:%M:%S))"
  ( source "$ENVF" && export $OVR && \
    W7_NODE_RANK=0 timeout "$TRY_TO" $PY "$P52/scripts/w7_2node.py" spec ) \
      > "$LOG" 2>&1
  nk=$(echo "$K" | awk -F, '{print NF}')
  nb=$(echo "$BATCH" | awk -F, '{print NF}')
  want=$((nk * nb))
  got=$(grep -cE "W7-2N K=.* batch=.* accept_len" "$LOG" 2>/dev/null)
  echo "[w67] $TAG try=$try -> ${got:-0}/$want rows ($(date +%H:%M:%S))"
  grep -hE "W7-2N K=|KV cache size|shared-KV|kv-window" "$LOG" 2>/dev/null | tail -8
  if [ "${got:-0}" -ge "$want" ]; then ok=1; break; fi
done
kill_stragglers
[ "$ok" = "1" ] || { echo "[w67] $TAG INCOMPLETE after $RETRIES tries"; exit 1; }
echo "[w67] $TAG done, profiles in $PROFDIR ($(date +%H:%M:%S))"
ls -1 "$PROFDIR"/self_spec_profile_*.json 2>/dev/null | wc -l
