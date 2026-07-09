#!/bin/bash
# Does the KV-quant verdict flip at 32k? At long ctx the draft is less
# fixed-overhead-bound and more KV-bound, so fp8 KV (global) MIGHT help the spec
# ratio more than at 16k (where nospec gained +12% vs spec +8% -> spec lost 0.95x).
# All 32k b8, NO window, clean. Anchor: nospec bf16 KV (~436). fp8 arms K=2 and K=3.
set -u
PHASE=/data/smcho/self-spec-moe/research/74_draft_step_cost
P52=/data/smcho/self-spec-moe/research/52_two_node_e2e
P57=/data/smcho/self-spec-moe/research/57_large_ep_spec_strategy
PY=/data/smcho/self-spec-moe/.venv/bin/python
ME="$(whoami)"
kill_mine(){ for p in 'w7_2node[.]py' 'EngineCor[e]' 'Worker_D[P]'; do pkill -9 -u "$ME" -f "$p" 2>/dev/null; done; sleep 5; }
run(){  # tag mode kvdtype K
  local TAG=$1 MODE=$2 KVD=$3 K=$4
  local LOG="$PHASE/logs/kvq32_${TAG}.log"
  local KVENV=""; [ -n "$KVD" ] && KVENV="W7_KV_CACHE_DTYPE=$KVD"
  kill_mine
  ( source "$PHASE/scripts/env_cloud4.sh"
    unset VLLM_SELF_SPEC_COMPILE_CONSISTENT
    export $KVENV
    export W7_KS=$K W7_BATCHES=8 W7_ITERS=4 W7_WARMUP=1 W7_TAG="p74_kvq32_${TAG}" \
      W7_MASTER_PORT=$((17900 + K + RANDOM%30)) W7_CTX_TOKENS=32768 W7_MAX_MODEL_LEN=36864 \
      W7_MAX_NUM_BATCHED=8192 W7_OUT="$PHASE/data" W7_PROMPT_FILE="$P57/data/prompts_ondist.txt" W7_CHAT=1
    W7_NODE_RANK=0 timeout 1400 "$PY" "$P52/scripts/w7_2node.py" "$MODE" ) > "$LOG" 2>&1
  echo "[kvq32] ${TAG}: $(grep -hE 'W7-2N.*batch=' "$LOG" | tail -1)"
  kill_mine
}
run nospec_bf16   nospec ""        1
run nospec_fp8    nospec fp8_e4m3  1
run spec_fp8_K2   spec   fp8_e4m3  2
run spec_fp8_K3   spec   fp8_e4m3  3
echo "[kvq32] DONE ($(date +%H:%M:%S))"
