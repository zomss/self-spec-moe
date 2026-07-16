#!/bin/bash
# The map-v6 measurement queue (emitted by the map's own uncertainty):
#   1. ngram e2e on V2-Lite (priced LCB ~2.45, e2e UNVALIDATED)
#   2. flr50+q_fp8 at moe b4/16k (priced on assumed ctx scaling)
# (vres needs lm_head-slice plumbing -- separate.)
# GPUs: MLA single-GPU arms on 6/7; MoE DP4 on 0-3.
set -u
PHASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$PHASE/../.." && pwd)"
P52="$REPO/research/52_two_node_e2e"
P57="$REPO/research/57_large_ep_spec_strategy"
PY="$REPO/.venv/bin/python"
mkdir -p "$PHASE/logs" "$PHASE/data/v6q"
ME="$(whoami)"
kill_mine(){ for p in 'w7_2node[.]py' 'EngineCor[e]' 'Worker_D[P]'; do pkill -9 -u "$ME" -f "$p" 2>/dev/null; done; sleep 5; }

mla(){  # arm mode K B gpu [extra-env...]
  local ARM=$1 MODE=$2 K=$3 B=$4 GPU=$5 CTX=16384; shift 5
  local TAG="v6q_mla_${ARM}_K${K}_b${B}_c16k"
  local LOG="$PHASE/logs/${TAG}.log"
  echo "[v6q] $TAG (GPU $GPU, $(date +%H:%M:%S))"
  ( export PATH="$REPO/.venv/bin:$PATH"
    export HF_HOME=/data/smcho/huggingface HF_HUB_OFFLINE=1 TMPDIR=/data/smcho/tmp
    export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
    export CUDA_VISIBLE_DEVICES="$GPU"
    export NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo
    unset VLLM_SELF_SPEC_COMPILE_CONSISTENT VLLM_SELF_SPEC_PROFILE
    export W7_MODEL=deepseek-ai/DeepSeek-V2-Lite W7_TP=1 W7_EP=0 W7_TRC=0 W7_EAGER=0
    export W7_NODES=1 W7_LOCAL_WORLD=1 W7_MASTER_IP=127.0.0.1
    export W7_GPU_MEM=0.90 W7_OUTLEN=160 W7_SHORTLEN=32
    export W7_CTX_TOKENS=$CTX W7_MAX_MODEL_LEN=$((CTX + 4096)) W7_MAX_NUM_BATCHED=8192
    export W7_KS=$K W7_BATCHES=$B W7_ITERS=8 W7_WARMUP=1
    export W7_TAG="$TAG" W7_OUT="$PHASE/data/v6q"
    export W7_PROMPT_FILE="$P57/data/prompts_ondist.txt" W7_CHAT=1
    export W7_MASTER_PORT=$((19400 + K + B + RANDOM % 40))
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=180
    for kv in "$@"; do export "${kv?}"; done
    W7_NODE_RANK=0 timeout 2400 "$PY" "$P52/scripts/w7_2node.py" "$MODE"
  ) > "$LOG" 2>&1
  grep -hE 'W7-2N.*batch=' "$LOG" | tail -1 | sed "s/^/       [$TAG] /"
}

FILTER="${1:-.}"
want(){ echo "$1" | grep -qE "$FILTER"; }
kill_mine
# parallel on GPUs 6/7 (single-GPU arms)
want mla_nospec && mla nospec8 nospec 0 32 6 &
want mla_ngram && mla ngram spec 4 32 7 W7_SPEC_METHOD=ngram &
wait
want mla_ngram_b8 && { mla nospec8b8 nospec 0 8 6 & mla ngramb8 spec 4 8 7 W7_SPEC_METHOD=ngram & wait; }
kill_mine
echo "[v6q] DONE ($(date +%H:%M:%S))"
