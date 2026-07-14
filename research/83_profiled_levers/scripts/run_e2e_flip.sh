#!/bin/bash
# E2b: e2e check of the four priced flips (MoE short-ctx OFF band -> flr).
# Arms per cell: nospec / lr-naive (EP-shard local route, the P24 lineage) /
# flr50 (FULL replica + freq resident sets, Phase-83 plumbing).
# Priced: flr50 1.09-1.11x at gamma2 vs naive ceiling 1.04-1.07.
# MoE Qwen3-30B-A3B DP4/EP4 on GPUs 0-3. Run BY PATH: bash scripts/run_e2e_flip.sh
set -u
PHASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$PHASE/../.." && pwd)"
P52="$REPO/research/52_two_node_e2e"
P57="$REPO/research/57_large_ep_spec_strategy"
P74="$REPO/research/74_draft_step_cost"
PY="$REPO/.venv/bin/python"
mkdir -p "$PHASE/logs" "$PHASE/data/e2e_flip"
ME="$(whoami)"
kill_mine(){ for p in 'w7_2node[.]py' 'EngineCor[e]' 'Worker_D[P]'; do pkill -9 -u "$ME" -f "$p" 2>/dev/null; done; sleep 5; }

run(){  # arm mode K B [extra-env...]
  local ARM=$1 MODE=$2 K=$3 B=$4 CTX=2048; shift 4
  local TAG="flip_${ARM}_K${K}_b${B}_c2k"
  local LOG="$PHASE/logs/${TAG}.log"
  echo "[flip] $TAG ($(date +%H:%M:%S))"
  kill_mine
  ( source "$P74/scripts/env_cloud4.sh"
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    export HF_HOME=/data/smcho/huggingface TMPDIR=/data/smcho/tmp
    unset VLLM_SELF_SPEC_COMPILE_CONSISTENT
    export W7_KS=$K W7_BATCHES=$B W7_ITERS=4 W7_WARMUP=1 W7_TAG="$TAG" \
      W7_MASTER_PORT=$((19100 + K + B + RANDOM % 40)) W7_CTX_TOKENS=$CTX \
      W7_MAX_MODEL_LEN=$((CTX + 4096)) W7_MAX_NUM_BATCHED=8192 \
      W7_OUT="$PHASE/data/e2e_flip" W7_PROMPT_FILE="$P57/data/prompts_ondist.txt" W7_CHAT=1
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=180
    for kv in "$@"; do export "${kv?}"; done
    W7_NODE_RANK=0 timeout 2400 "$PY" "$P52/scripts/w7_2node.py" "$MODE"
  ) > "$LOG" 2>&1
  grep -hE 'W7-2N.*batch=' "$LOG" | tail -1 | sed "s/^/       [$TAG] /"
  kill_mine
}

FILTER="${1:-.}"
want(){ echo "$1" | grep -qE "$FILTER"; }
for B in 4 8 32; do
  want "nospec_b$B" && run nospec nospec 0 $B
  want "lrnaive_b$B" && run lrnaive spec 2 $B W7_DRAFT_LOCAL_ROUTE=1
  want "flr50_b$B" && run flr50 spec 2 $B W7_DRAFT_LOCAL_ROUTE=1 \
    W7_DRAFT_FULL_REPLICA=1 \
    VLLM_SELF_SPEC_DRAFT_RESIDENT_SETS="$PHASE/data/resident_sets_flr50.pt"
done
kill_mine
echo "[flip] DONE ($(date +%H:%M:%S))"
