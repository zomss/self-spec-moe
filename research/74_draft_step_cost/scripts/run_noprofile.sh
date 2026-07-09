#!/bin/bash
# Is the "between-forward overhead" real or a profiler artifact? Measure spec
# throughput WITHOUT VLLM_SELF_SPEC_PROFILE (no per-region cuda.synchronize),
# batch-invariant OFF. base + window at 16k/32k b8. Compare to clean no-spec
# (599 @16k b8, 436 @32k b8). Prior profiled spec: base 370/311, window 354/386.
set -u
PHASE=/data/smcho/self-spec-moe/research/74_draft_step_cost
P52=/data/smcho/self-spec-moe/research/52_two_node_e2e
P57=/data/smcho/self-spec-moe/research/57_large_ep_spec_strategy
PY=/data/smcho/self-spec-moe/.venv/bin/python
ME="$(whoami)"
kill_mine(){ for p in 'w7_2node[.]py' 'EngineCor[e]' 'Worker_D[P]'; do pkill -9 -u "$ME" -f "$p" 2>/dev/null; done; sleep 5; }
for CTX in 16384 32768; do
  MAXLEN=$((CTX+4096))
  for ARM in base window; do
    DELTA=""; [ "$ARM" = window ] && DELTA="VLLM_SELF_SPEC_DRAFT_KV_WINDOW=512 VLLM_SELF_SPEC_DRAFT_KV_SINKS=16"
    LOG="$PHASE/logs/np_${ARM}_ctx${CTX}_b8.log"
    kill_mine
    ( source "$PHASE/scripts/env_cloud4.sh"
      unset VLLM_SELF_SPEC_COMPILE_CONSISTENT          # batch-invariant OFF
      # NOTE: no VLLM_SELF_SPEC_PROFILE -> no per-region syncs
      export W7_KS=4 W7_BATCHES=8 W7_ITERS=3 W7_WARMUP=1 W7_TAG="p74_np_${ARM}_ctx${CTX}_b8" \
        W7_MASTER_PORT=$((17000 + CTX/1000 + RANDOM%40)) W7_CTX_TOKENS="$CTX" W7_MAX_MODEL_LEN="$MAXLEN" \
        W7_MAX_NUM_BATCHED=8192 W7_OUT="$PHASE/data" W7_PROMPT_FILE="$P57/data/prompts_ondist.txt" W7_CHAT=1 \
        $DELTA
      W7_NODE_RANK=0 timeout 1200 "$PY" "$P52/scripts/w7_2node.py" spec ) > "$LOG" 2>&1
    echo "[np] ${ARM} ctx${CTX} b8: $(grep -hE 'W7-2N.*batch=' "$LOG" | tail -1)"
    kill_mine
  done
done
echo "[np] DONE ($(date +%H:%M:%S))"
