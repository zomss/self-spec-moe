#!/bin/bash
# Phase 74 profiling: DRAFT-step vs VERIFY-step time across batch x context, with
# a no-spec forward reference to test whether the draft is "fully optimized".
#   draft_forward (spec, B*1 tok, EAGER chain) vs nospec_forward (B*1 tok,
#   CUDA-GRAPHED) -- same bf16 weights, same tokens -> the gap is pure draft
#   overhead. verify (spec, B*(K+1) tok) is the target forward (graphed).
# For each context: ONE nospec engine (batch list -> per-batch tok/s -> fwd
# time) + ONE spec-base engine PER batch (clean per-batch profiler regions).
# bf16 EP-routed base draft (env_cloud4.sh defaults; no quant/window/local-route
# -> the cleanest eager-overhead reference). Cleanup scoped to THIS user.
set -u
PHASE=/data/smcho/self-spec-moe/research/74_draft_step_cost
P52=/data/smcho/self-spec-moe/research/52_two_node_e2e
P57=/data/smcho/self-spec-moe/research/57_large_ep_spec_strategy
PY=/data/smcho/self-spec-moe/.venv/bin/python
ENVF="$PHASE/scripts/env_cloud4.sh"
ME="$(whoami)"
PROMPTS="$P57/data/prompts_ondist.txt"
K=4; ITERS=2; WARMUP=1; PWARMUP=20   # profiler regions avg over 100s of steps; ITERS=2 keeps the grid fast
CONTEXTS="${CONTEXTS:-2048 16384 32768}"
BATCHES="${BATCHES:-2 4 8}"
mkdir -p "$PHASE/logs" "$PHASE/data"

kill_mine(){ for p in 'w7_2node[.]py' 'EngineCor[e]' 'Worker_D[P]'; do pkill -9 -u "$ME" -f "$p" 2>/dev/null; done; sleep 5; }

run_one(){  # mode ctx batch tag_extra profdir
  local MODE=$1 CTX=$2 BATCH=$3 TAG=$4 PROFDIR=$5
  local MAXLEN=$((CTX + 4096))
  local LOG="$PHASE/logs/prof_${TAG}.log"
  local PROF=""
  [ -n "$PROFDIR" ] && { mkdir -p "$PROFDIR"; rm -f "$PROFDIR"/*.json 2>/dev/null; \
    PROF="VLLM_SELF_SPEC_PROFILE=1 VLLM_SELF_SPEC_PROFILE_OUT=$PROFDIR VLLM_SELF_SPEC_PROFILE_WARMUP=$PWARMUP"; }
  local PORT=$((15400 + RANDOM % 300))
  kill_mine
  local OVR="W7_KS=$K W7_BATCHES=$BATCH W7_ITERS=$ITERS W7_WARMUP=$WARMUP \
W7_TAG=p74_${TAG} W7_MASTER_PORT=$PORT W7_CTX_TOKENS=$CTX W7_MAX_MODEL_LEN=$MAXLEN \
W7_MAX_NUM_BATCHED=8192 W7_OUT=$PHASE/data W7_PROMPT_FILE=$PROMPTS W7_CHAT=1 $PROF"
  echo "[prof] $TAG mode=$MODE ($(date +%H:%M:%S))"
  ( source "$ENVF" && export $OVR && \
    W7_NODE_RANK=0 timeout 1500 $PY "$P52/scripts/w7_2node.py" "$MODE" ) > "$LOG" 2>&1
  grep -hE "W7-2N.*batch=|GPU KV cache size" "$LOG" | tail -4
  kill_mine
}

for CTX in $CONTEXTS; do
  # no-spec: one engine, all batches (per-batch tok/s -> forward time)
  run_one nospec "$CTX" "$(echo $BATCHES | tr ' ' ',')" "nospec_ctx${CTX}" ""
  # spec base: one engine per batch (clean per-batch draft/verify profiler)
  for B in $BATCHES; do
    run_one spec "$CTX" "$B" "base_ctx${CTX}_b${B}" "$PHASE/data/prof_base_ctx${CTX}_b${B}"
  done
done
echo "[prof] COMPLETE ($(date +%H:%M:%S))"
