#!/bin/bash
# E2 controls only (the broken asym arms hang the full sweep on timeout). Two arms:
#   greedy (T=0) -> isolates the temperature effect vs the T=1.0 primary
#   SHARED_KV=1  -> expect INFLATED tau (draft attends target's exact KV)
# W4-sym, gamma=5, b8, ctx2k. Short timeout so a failure can't hang 40 min. Run BY PATH.
set -u
PHASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$PHASE/../.." && pwd)"
P52="$REPO/research/52_two_node_e2e"; PY="$REPO/.venv/bin/python"
CKPT_ROOT="${CKPT_ROOT:-/data/smcho/ckpts}"
W4="$CKPT_ROOT/Qwen2.5-7B-Instruct-W4A16-INT4-sym"
PROMPTS="$PHASE/prompts/math_prompts.txt"
ME="$(whoami)"
kill_mine(){ for p in 'w7_2node[.]py' 'EngineCor[e]'; do pkill -9 -u "$ME" -f "$p" 2>/dev/null; done; sleep 4; }
tau(){ local tag=$1 t=$2 skv=$3
  local LOG="$PHASE/logs/e2c_${tag}.log"; kill_mine
  ( source "$PHASE/scripts/env_e75.sh"
    export E75_SHARED_KV="$skv" VLLM_SELF_SPEC_SHARED_KV="$skv"
    export W7_SPEC_MODEL="$W4" W7_KS=5 W7_TEMP="$t" W7_TOPP=1.0
    export W7_BATCHES=8 W7_ITERS=2 W7_WARMUP=1 W7_TAG="p75_e2c_${tag}"
    export W7_CTX_TOKENS=2048 W7_MAX_MODEL_LEN=3072
    export W7_PROMPT_FILE="$PROMPTS" W7_CHAT=1 W7_PROMPT_OFFSET=0
    export W7_OUT="$PHASE/data" W7_MASTER_PORT=$((19300 + RANDOM % 40))
    W7_NODE_RANK=0 timeout 700 "$PY" "$P52/scripts/w7_2node.py" spec ) > "$LOG" 2>&1
  echo "[e2c] ${tag}: $(grep -hE 'W7-2N K=' "$LOG" | tail -1)"
  kill_mine
}
tau greedy_T0     0.0 0    # greedy control (temperature de-risk)
tau sharedkv_T1   1.0 1    # SHARED_KV=1 (expect inflated tau)
echo "[e2c] DONE ($(date +%H:%M:%S))  primary T=1.0 ref: w4sym g5 = 4.708"
