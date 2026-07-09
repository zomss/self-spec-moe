#!/bin/bash
# Phase 74: draft-step-cost lever stack on THIS box (single NVLink node, GPUs
# 4-7 = DP4/EP4). Cumulative arms (NO local routing -- that is a large-EP
# lever, deferred):
#   base   = EP-routed bf16 self-draft, full KV        (env_cloud4.sh defaults)
#   window = base + sparse (sinks+window) draft attention (DRAFT_KV_WINDOW=512)
#   kvq    = window + fp8 KV-cache quant (W7_KV_CACHE_DTYPE=fp8)  [bounded-lossy]
#
# One engine launch per (arm,ctx,batch) POINT so the SelfSpecProfiler regions
# (draft_forward / draft_forward_first / verify / draft_chain) are not mixed
# across batches. Per-rank decode batch == W7_BATCHES (replicated-batch DP,
# Phase 73). Cleanup is scoped to THIS user -- root's GPU 0/1 engines are never
# touched, and we are isolated to 4-7 via CUDA_VISIBLE_DEVICES.
#
# Usage:
#   run_stack.sh                              # default 16k base/window/kvq @ b8,b16
#   ARMS="kvq" POINTS="16384:24 16384:32" run_stack.sh    # probe raised ceiling
#   ARMS="base" POINTS="2048:1 2048:2 2048:4 2048:8 2048:16" run_stack.sh  # F_fixed/m
set -u
PHASE=/data/smcho/self-spec-moe/research/74_draft_step_cost
P52=/data/smcho/self-spec-moe/research/52_two_node_e2e
P57=/data/smcho/self-spec-moe/research/57_large_ep_spec_strategy
PY=/data/smcho/self-spec-moe/.venv/bin/python
ENVF="$PHASE/scripts/env_cloud4.sh"
ME="$(whoami)"

K="${K:-4}"
ITERS="${ITERS:-2}"; WARMUP="${WARMUP:-1}"
PWARMUP="${PWARMUP:-20}"
TRY_TO="${TRY_TO:-1500}"; RETRIES="${RETRIES:-2}"
PROMPTS="${PROMPTS:-$P57/data/prompts_ondist.txt}"
ARMS="${ARMS:-base window kvq}"
POINTS="${POINTS:-16384:8 16384:16}"
mkdir -p "$PHASE/logs" "$PHASE/data"

# Per-arm env delta (cumulative). Echoed into the launch env.
arm_env() {
  case "$1" in
    base)   echo "" ;;
    window) echo "VLLM_SELF_SPEC_DRAFT_KV_WINDOW=512 VLLM_SELF_SPEC_DRAFT_KV_SINKS=16 W7_KV_WINDOW_DEBUG=1" ;;
    kvq)    echo "VLLM_SELF_SPEC_DRAFT_KV_WINDOW=512 VLLM_SELF_SPEC_DRAFT_KV_SINKS=16 W7_KV_WINDOW_DEBUG=1 W7_KV_CACHE_DTYPE=fp8" ;;
    # fp8 EP-routed draft (replica=0, local_route=0 from env) -- draft weights fp8
    # (2x less weight traffic) vs the bf16 verify. The real "D < V" test.
    fp8d)   echo "W7_DRAFT_QUANT=fp8" ;;
    fp8dw)  echo "W7_DRAFT_QUANT=fp8 VLLM_SELF_SPEC_DRAFT_KV_WINDOW=512 VLLM_SELF_SPEC_DRAFT_KV_SINKS=16 W7_KV_WINDOW_DEBUG=1" ;;
    # fp8 draft + fp8 KV (doubles pool so high batch fits despite the +fp8-copy HBM)
    fp8dwq) echo "W7_DRAFT_QUANT=fp8 VLLM_SELF_SPEC_DRAFT_KV_WINDOW=512 VLLM_SELF_SPEC_DRAFT_KV_SINKS=16 W7_KV_WINDOW_DEBUG=1 W7_KV_CACHE_DTYPE=fp8" ;;
    *) echo "BAD_ARM"; return 1 ;;
  esac
}

# Kill ONLY our own stragglers (never root's GPU0/1 engines).
kill_mine() {
  for pat in 'w7_2node[.]py' 'EngineCor[e]' 'Worker_D[P]'; do
    pkill -9 -u "$ME" -f "$pat" 2>/dev/null
  done
  sleep 5
}

port=14710
for arm in $ARMS; do
  ADELTA="$(arm_env "$arm")" || { echo "[w74] bad arm $arm"; exit 2; }
  for pt in $POINTS; do
    CTX="${pt%%:*}"; BATCH="${pt##*:}"
    MAXLEN=$((CTX < 4096 ? 4096 : CTX + 4096))
    TAG="${arm}_ctx${CTX}_b${BATCH}"
    PROFDIR="$PHASE/data/prof_${TAG}"
    mkdir -p "$PROFDIR"; rm -f "$PROFDIR"/self_spec_profile_*.json 2>/dev/null
    kill_mine

    ok=0
    for try in $(seq 1 "$RETRIES"); do
      port=$((port + 3))
      LOG="$PHASE/logs/${TAG}_try${try}.log"
      OVR="W7_KS=$K W7_BATCHES=$BATCH W7_ITERS=$ITERS W7_WARMUP=$WARMUP \
W7_TAG=p74_${TAG} W7_MASTER_PORT=$port W7_CTX_TOKENS=$CTX W7_MAX_MODEL_LEN=$MAXLEN \
W7_MAX_NUM_BATCHED=8192 W7_OUT=$PHASE/data W7_PROMPT_FILE=$PROMPTS W7_CHAT=1 \
$ADELTA VLLM_SELF_SPEC_PROFILE=1 VLLM_SELF_SPEC_PROFILE_OUT=$PROFDIR \
VLLM_SELF_SPEC_PROFILE_WARMUP=$PWARMUP"
      echo "[w74] $TAG try=$try to=${TRY_TO}s ($(date +%H:%M:%S))"
      ( source "$ENVF" && export $OVR && \
        W7_NODE_RANK=0 timeout "$TRY_TO" $PY "$P52/scripts/w7_2node.py" spec ) \
          > "$LOG" 2>&1
      got=$(grep -cE "W7-2N K=.* batch=.* accept_len" "$LOG" 2>/dev/null)
      nprof=$(ls -1 "$PROFDIR"/self_spec_profile_*.json 2>/dev/null | wc -l)
      echo "[w74] $TAG try=$try -> ${got:-0} rows, ${nprof} prof ($(date +%H:%M:%S))"
      grep -hE "W7-2N K=|KV cache size|GPU KV|kv-window|shared-KV|kv_cache_dtype|Bounded" "$LOG" 2>/dev/null | tail -6
      if [ "${got:-0}" -ge 1 ] && [ "$nprof" -ge 1 ]; then ok=1; break; fi
    done
    kill_mine
    [ "$ok" = 1 ] && echo "[w74] $TAG DONE" || echo "[w74] $TAG INCOMPLETE -> next"
  done
done
kill_mine
echo "[w74] STACK COMPLETE ($(date +%H:%M:%S))"
