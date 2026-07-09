#!/bin/bash
# Phase 74 follow-up: the missing no-spec DENOMINATORS (at both KV precisions,
# for a fair comparison) + the high-batch CEILING probe. Runs after the core
# sweep. Each line is "arm ctx batch"; arm sets mode + env delta:
#   nospec_bf16 : plain decode, bf16 KV      (denominator for base/window)
#   nospec_fp8  : plain decode, fp8 KV       (fair denominator for kvq)
#   kvq         : spec, window + fp8 KV      (residency+amortization win)
#   base        : spec, bf16 full KV         (bf16 ceiling / livelock evidence)
# One engine per point; cleanup scoped to THIS user (root's GPU0/1 untouched).
set -u
PHASE=/data/smcho/self-spec-moe/research/74_draft_step_cost
P52=/data/smcho/self-spec-moe/research/52_two_node_e2e
P57=/data/smcho/self-spec-moe/research/57_large_ep_spec_strategy
PY=/data/smcho/self-spec-moe/.venv/bin/python
ENVF="$PHASE/scripts/env_cloud4.sh"
ME="$(whoami)"
PROMPTS="$P57/data/prompts_ondist.txt"
K=4; ITERS=2; WARMUP=1; PWARMUP=20
mkdir -p "$PHASE/logs" "$PHASE/data"

# arm -> "mode|env delta"
arm_spec() {
  case "$1" in
    nospec_bf16) echo "nospec|" ;;
    nospec_fp8)  echo "nospec|W7_KV_CACHE_DTYPE=fp8" ;;
    kvq)         echo "spec|VLLM_SELF_SPEC_DRAFT_KV_WINDOW=512 VLLM_SELF_SPEC_DRAFT_KV_SINKS=16 W7_KV_WINDOW_DEBUG=1 W7_KV_CACHE_DTYPE=fp8" ;;
    base)        echo "spec|" ;;
    *) echo "BAD|"; return 1 ;;
  esac
}

kill_mine() { for p in 'w7_2node[.]py' 'EngineCor[e]' 'Worker_D[P]'; do pkill -9 -u "$ME" -f "$p" 2>/dev/null; done; sleep 5; }

# Sequence: denominators first (all fit at b8/b16; nospec_fp8 also b32),
# then the spec high-batch points. base b24 has a SHORT timeout (expected
# to livelock at the ~b18 bf16 pool -> that IS the ceiling data point).
SEQ="nospec_bf16:16384:8 nospec_bf16:16384:16 \
nospec_fp8:16384:8 nospec_fp8:16384:16 nospec_fp8:16384:32 \
kvq:16384:24 kvq:16384:32 base:16384:24"

port=15010
for item in $SEQ; do
  arm="${item%%:*}"; rest="${item#*:}"; CTX="${rest%%:*}"; BATCH="${rest##*:}"
  spec="$(arm_spec "$arm")"; MODE="${spec%%|*}"; ADELTA="${spec#*|}"
  MAXLEN=$((CTX + 4096))
  TAG="${arm}_ctx${CTX}_b${BATCH}"
  # base high-batch is expected to livelock -> bound it tight so it doesn't eat 20m
  TO=1200; [ "$arm" = base ] && [ "$BATCH" -ge 24 ] && TO=500
  PROFDIR="$PHASE/data/prof_${TAG}"; mkdir -p "$PROFDIR"
  rm -f "$PROFDIR"/self_spec_profile_*.json 2>/dev/null
  kill_mine
  port=$((port + 3))
  LOG="$PHASE/logs/${TAG}.log"
  OVR="W7_KS=$K W7_BATCHES=$BATCH W7_ITERS=$ITERS W7_WARMUP=$WARMUP \
W7_TAG=p74_${TAG} W7_MASTER_PORT=$port W7_CTX_TOKENS=$CTX W7_MAX_MODEL_LEN=$MAXLEN \
W7_MAX_NUM_BATCHED=8192 W7_OUT=$PHASE/data W7_PROMPT_FILE=$PROMPTS W7_CHAT=1 \
$ADELTA VLLM_SELF_SPEC_PROFILE=1 VLLM_SELF_SPEC_PROFILE_OUT=$PROFDIR \
VLLM_SELF_SPEC_PROFILE_WARMUP=$PWARMUP"
  echo "[w74f] $TAG mode=$MODE to=${TO}s ($(date +%H:%M:%S))"
  ( source "$ENVF" && export $OVR && \
    W7_NODE_RANK=0 timeout "$TO" $PY "$P52/scripts/w7_2node.py" "$MODE" ) \
      > "$LOG" 2>&1
  got=$(grep -cE "W7-2N.*batch=.*tok/s" "$LOG" 2>/dev/null)
  echo "[w74f] $TAG -> ${got:-0} rows ($(date +%H:%M:%S))"
  grep -hE "W7-2N.*batch=|GPU KV cache size|Maximum concurrency|preempt|livelock" "$LOG" 2>/dev/null | tail -5
  kill_mine
done
echo "[w74f] FOLLOWUP COMPLETE ($(date +%H:%M:%S))"
