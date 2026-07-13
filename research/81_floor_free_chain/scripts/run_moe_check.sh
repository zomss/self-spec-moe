#!/bin/bash
# Phase 81 MoE no-regression check (README E2): the E1b/E2b vLLM changes are
# env-gated OFF on the default MoE path -- verify the P74 MoE window cells
# reproduce, then one exploratory arm with the fixed-chain stack ON.
# P74 reference (b8/16k, Qwen3-30B-A3B DP4/EP4, window 512/16):
#   K=2: 692.8 +-7.9 tok/s, accept 2.905
#   K=3: 638.9 +-27.5 tok/s, accept 3.824
# GPUs 0-3 (user-cleared 2026-07-13). Run BY PATH: bash scripts/run_moe_check.sh
set -u
PHASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$PHASE/../.." && pwd)"
P52="$REPO/research/52_two_node_e2e"
P57="$REPO/research/57_large_ep_spec_strategy"
P74="$REPO/research/74_draft_step_cost"
PY="$REPO/.venv/bin/python"
mkdir -p "$PHASE/logs" "$PHASE/data/moe_check"
ME="$(whoami)"
kill_mine(){ for p in 'w7_2node[.]py' 'EngineCor[e]' 'Worker_D[P]'; do pkill -9 -u "$ME" -f "$p" 2>/dev/null; done; sleep 5; }

run(){  # arm K [extra-env...]
  local ARM=$1 K=$2 CTX=16384; shift 2
  local TAG="moechk_${ARM}_K${K}_b8_c16k"
  local LOG="$PHASE/logs/${TAG}.log"
  echo "[moechk] $TAG ($(date +%H:%M:%S))"
  kill_mine
  ( source "$P74/scripts/env_cloud4.sh"
    # box deltas since P74: GPUs 0-3 now; caches on /data
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    export HF_HOME=/data/smcho/huggingface TMPDIR=/data/smcho/tmp
    unset VLLM_SELF_SPEC_COMPILE_CONSISTENT
    export W7_KS=$K W7_BATCHES=8 W7_ITERS=4 W7_WARMUP=1 W7_TAG="$TAG" \
      W7_MASTER_PORT=$((18500 + K + RANDOM % 40)) W7_CTX_TOKENS=$CTX \
      W7_MAX_MODEL_LEN=$((CTX + 4096)) W7_MAX_NUM_BATCHED=8192 \
      W7_OUT="$PHASE/data/moe_check" W7_PROMPT_FILE="$P57/data/prompts_ondist.txt" W7_CHAT=1
    export VLLM_SELF_SPEC_DRAFT_KV_WINDOW=512 VLLM_SELF_SPEC_DRAFT_KV_SINKS=16
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=180
    for kv in "$@"; do export "${kv?}"; done
    W7_NODE_RANK=0 timeout 2400 "$PY" "$P52/scripts/w7_2node.py" spec
  ) > "$LOG" 2>&1
  grep -hE 'W7-2N.*batch=' "$LOG" | tail -1 | sed "s/^/       [$TAG] /"
  kill_mine
}

FILTER="${1:-.}"
want(){ echo "$1" | grep -qE "$FILTER"; }
want noreg2 && run noreg 2
want noreg3 && run noreg 3
# exploratory: fixed-chain stack on the MoE EP-routed draft
want fixed3 && run fixed 3 VLLM_SELF_SPEC_DRAFT_FULLCG=1 VLLM_SELF_SPEC_DRAFT_STEP0_FULL_CG=1 \
  VLLM_SELF_SPEC_CPU_ORCH=1 W7_ASYNC_SCHED=1 VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE=1
kill_mine
echo "[moechk] DONE ($(date +%H:%M:%S))"
