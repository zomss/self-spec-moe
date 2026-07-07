#!/bin/bash
# Phase 73: draft-forward overhead decomposition. Batch-scaling sweep of the
# PIECEWISE fp8-full-replica shared-KV W512 K4 draft on ONE node (h106,
# DP8/EP8). One engine launch per (ctx,batch) POINT so the SelfSpecProfiler
# region samples (draft_forward / draft_forward_first / verify / draft_chain)
# are NOT mixed across batches -- the profiler accumulates per-process and does
# not reset between batches, so each batch needs its own PROFILE_OUT dir.
#
# Per-rank decode batch = W7_BATCHES / DP(8)  (P72-anchored: W7_BATCHES=8 ->
# num_reqs=1/rank, draft ~11 ms b1). Short 2k ctx lets per-rank batch scale to
# 16 free of the 16k KV-pool cap; ONE 16k b8 point checks ctx-invariance.
#
# MUST run ON h106 (shared node). Before every launch: STOP if any non-
# v-sukmincho GPU compute process appears (beyond the permanent stale ctx);
# cleanup only OUR patterns; never sudo, never kill others.
#
# Usage:  run_sweep.sh            # full sweep (default POINTS)
#         POINTS="2048:8 2048:16" run_sweep.sh   # custom
set -u
PHASE=/h/v-sukmincho/self-spec-moe/research/73_overhead_decomp
P52=/h/v-sukmincho/self-spec-moe/research/52_two_node_e2e
P57=/h/v-sukmincho/self-spec-moe/research/57_large_ep_spec_strategy
PY=/h/v-sukmincho/self-spec-moe/.venv/bin/python
ENVF="$PHASE/scripts/env_h106.sh"
K="${K:-4}"
ITERS="${ITERS:-2}"; WARMUP="${WARMUP:-1}"
PWARMUP="${PWARMUP:-40}"
TRY_TO="${TRY_TO:-2000}"; RETRIES="${RETRIES:-3}"
PROMPTS="${PROMPTS:-$P57/data/prompts_ondist.txt}"
# ctx:batch points. per-rank decode batch == b (replicated-batch DP; MEASURED,
# see results.md). Full curve b1..64 at 2k + one 16k b8 ctx-invariance xcheck.
# (b128 @ 2k = 262k tok > 239k pool -> thrash; omitted.)
POINTS="${POINTS:-2048:1 2048:2 2048:4 2048:8 2048:16 2048:32 2048:64 16384:8}"
mkdir -p "$PHASE/logs" "$PHASE/data"

ARMB="W7_DRAFT_QUANT=fp8 W7_DRAFT_FULL_REPLICA=1 W7_DRAFT_LOCAL_ROUTE=1 \
W7_DRAFT_NODE_LOCAL=0 \
VLLM_SELF_SPEC_SHARED_KV=1 VLLM_SELF_SPEC_DRAFT_KV_WINDOW=512 \
VLLM_SELF_SPEC_DRAFT_KV_SINKS=16 W7_KV_WINDOW_DEBUG=1 \
VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1 \
VLLM_SELF_SPEC_DRAFT_SKIP_DP_COORD=1"

kill_stragglers() {
  for pat in 'w7_2node[.]py' 'EngineCor[e]' 'Worker_D[P]'; do
    pkill -9 -f "$pat" 2>/dev/null
  done
  sleep 6
}
# STRICT shared-node guard: yield if ANY non-v-sukmincho GPU compute app is
# present (the permanent ~522 MiB stale ctx pid has no owner -> ignored).
foreign_gpu_busy() {
  local pids p u me; me="$(whoami)"
  pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
  for p in $pids; do
    u="$(ps -o user= -p "$p" 2>/dev/null | tr -d ' ')"
    if [ -n "$u" ] && [ "$u" != "$me" ]; then
      echo "[w73] FOREIGN GPU pid=$p user=$u -> h106 contended, yielding"
      return 0
    fi
  done
  return 1
}

port=13940
for pt in $POINTS; do
  CTX="${pt%%:*}"; BATCH="${pt##*:}"
  MAXLEN=$((CTX < 4096 ? 4096 : CTX + 4096))
  TAG="ctx${CTX}_b${BATCH}"
  PROFDIR="$PHASE/data/prof_${TAG}"
  mkdir -p "$PROFDIR"; rm -f "$PROFDIR"/self_spec_profile_*.json 2>/dev/null

  if foreign_gpu_busy; then
    echo "[w73] ABORT sweep at $TAG (foreign GPU). Partial results kept."
    exit 3
  fi
  kill_stragglers

  ok=0
  for try in $(seq 1 "$RETRIES"); do
    if foreign_gpu_busy; then
      echo "[w73] ABORT (foreign GPU appeared mid-sweep)"; exit 3
    fi
    port=$((port + 5))
    LOG="$PHASE/logs/${TAG}_try${try}.log"
    OVR="W7_NODES=1 W7_LOCAL_WORLD=8 W7_KS=$K W7_BATCHES=$BATCH \
W7_ITERS=$ITERS W7_WARMUP=$WARMUP W7_GPU_MEM=0.90 W7_TAG=p73_${TAG} \
W7_MASTER_PORT=$port W7_CTX_TOKENS=$CTX W7_MAX_MODEL_LEN=$MAXLEN \
W7_MAX_NUM_BATCHED=8192 W7_OUT=$PHASE/data W7_PROMPT_FILE=$PROMPTS W7_CHAT=1 \
$ARMB VLLM_SELF_SPEC_PROFILE=1 VLLM_SELF_SPEC_PROFILE_OUT=$PROFDIR \
VLLM_SELF_SPEC_PROFILE_WARMUP=$PWARMUP"
    echo "[w73] $TAG K=$K b=$BATCH ctx=$CTX try=$try to=${TRY_TO}s ($(date +%H:%M:%S))"
    ( source "$ENVF" && export $OVR && \
      W7_NODE_RANK=0 timeout "$TRY_TO" $PY "$P52/scripts/w7_2node.py" spec ) \
        > "$LOG" 2>&1
    got=$(grep -cE "W7-2N K=.* batch=.* accept_len" "$LOG" 2>/dev/null)
    echo "[w73] $TAG try=$try -> ${got:-0}/1 rows ($(date +%H:%M:%S))"
    grep -hE "W7-2N K=|W7-2N ctx|KV cache size|shared-KV|kv-window|GPU KV" "$LOG" 2>/dev/null | tail -6
    nprof=$(ls -1 "$PROFDIR"/self_spec_profile_*.json 2>/dev/null | wc -l)
    if [ "${got:-0}" -ge 1 ] && [ "$nprof" -ge 1 ]; then ok=1; break; fi
  done
  kill_stragglers
  if [ "$ok" != "1" ]; then
    echo "[w73] $TAG INCOMPLETE after $RETRIES tries -- continuing to next point"
  else
    echo "[w73] $TAG done, $(ls -1 "$PROFDIR"/self_spec_profile_*.json | wc -l) prof dumps ($(date +%H:%M:%S))"
  fi
done
kill_stragglers
echo "[w73] SWEEP COMPLETE ($(date +%H:%M:%S))"
