#!/bin/bash
# Phase 72: single-node (h107, DP8/EP8) torch-profiler KERNEL trace of the
# Phase-70 full-CG draft arm. Reuses the P65/P64 trace harness (w7_trace16k.py,
# rank-0 torch profiler) with the P70 arm-B env + VLLM_SELF_SPEC_DRAFT_FULLCG=1
# so rank 0 records the FULL-CG draft-chain REPLAY at 16k / W512 / cap544.
#
# Usage: run_trace.sh [TAG] [K] [BATCH] [TRACE_LEN] [PORT] [TRY_TO]
set -u
TAG="${1:-fullcg}"; K="${2:-4}"; BATCH="${3:-8}"; TRACE_LEN="${4:-40}"
PORT="${5:-17700}"; TRY_TO="${6:-2000}"
PHASE=/h/v-sukmincho/self-spec-moe/research/72_kernel_trace
P65=/h/v-sukmincho/self-spec-moe/research/65_draft_overhead_opt
P70=/h/v-sukmincho/self-spec-moe/research/70_fullcg_coverage
P57=/h/v-sukmincho/self-spec-moe/research/57_large_ep_spec_strategy
PY=/h/v-sukmincho/self-spec-moe/.venv/bin/python
ENVF="$P70/scripts/env_1node.sh"
TRACE_DIR="$PHASE/data/trace_${TAG}_w512k${K}_b${BATCH}"
LOG="$PHASE/logs/trace_${TAG}_w512k${K}_b${BATCH}.log"
mkdir -p "$PHASE/logs" "$PHASE/data" "$TRACE_DIR"
rm -f "$TRACE_DIR"/*.pt.trace.json* 2>/dev/null

# P70 arm-B: fp8 comm-free replica + shared-KV + W512 sinks16 + P65 flag stack.
ARMB="W7_DRAFT_QUANT=fp8 W7_DRAFT_FULL_REPLICA=1 W7_DRAFT_LOCAL_ROUTE=1 \
W7_DRAFT_NODE_LOCAL=0 \
VLLM_SELF_SPEC_SHARED_KV=1 VLLM_SELF_SPEC_DRAFT_KV_WINDOW=512 \
VLLM_SELF_SPEC_DRAFT_KV_SINKS=16 W7_KV_WINDOW_DEBUG=1 \
VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1 \
VLLM_SELF_SPEC_DRAFT_SKIP_DP_COORD=1"
# Full-CG replay arm + descriptor debug + profiler custom scopes.
FULLCG="VLLM_SELF_SPEC_DRAFT_FULLCG=1 W7_FULLCG_DBG=1 \
VLLM_CUSTOM_SCOPES_FOR_PROFILING=1"

kill_stragglers() {
  for pat in 'w7_trace16k[.]py' 'w7_2node[.]py' 'EngineCor[e]' 'Worker_D[P]'; do
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
      echo "[w72] foreign GPU pid=$p user=$u"; return 0
    fi
  done
  return 1
}

while foreign_gpu_busy; do
  echo "[w72] waiting 120s for foreign GPU ($(date +%H:%M:%S))"; sleep 120
done
kill_stragglers

RUN="W7_NODES=1 W7_LOCAL_WORLD=8 W7_K=$K W7_BATCH=$BATCH \
W7_TRACE_LEN=$TRACE_LEN W7_TRACE_DIR=$TRACE_DIR \
W7_CTX_TOKENS=16384 W7_MAX_MODEL_LEN=20480 W7_MAX_NUM_BATCHED=8192 \
W7_GPU_MEM=0.90 W7_MASTER_PORT=$PORT W7_CHAT=1 \
W7_PROMPT_FILE=$P57/data/prompts_ondist.txt \
$ARMB $FULLCG"

echo "[w72] $TAG K=$K b=$BATCH len=$TRACE_LEN to=${TRY_TO}s ($(date +%H:%M:%S))"
( source "$ENVF" && export $RUN && \
  W7_NODE_RANK=0 timeout "$TRY_TO" $PY "$P65/scripts/w7_trace16k.py" ) \
    > "$LOG" 2>&1
RC=$?
kill_stragglers
echo "[w72] === fullcg-dbg / kv-window / TRACE16K lines ==="
grep -hE "fullcg-dbg|FULL-CG|REPLAY|CAPTURE-MISS|PASSTHROUGH|kv-window|shared-KV|TRACE16K|scratchpad" "$LOG" 2>/dev/null | sort -u | tail -30
echo "[w72] === trace files ==="
ls -lh "$TRACE_DIR" 2>/dev/null | tail -5
echo "[w72] $TAG done RC=$RC ($(date +%H:%M:%S))"
