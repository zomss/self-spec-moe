#!/usr/bin/env bash
# W2: dual-currency regime sweep. dense + llama x {off, 512, 2048}, seed 0,
# all runs NOTUNE (W1 source-A decision). Pinning NOT adopted (P-W1b1
# refuted -- neither helps nor hurts; W2_PIN=1 re-enables for forensics).
# ITERS=4 so a 1-2 round slow-mode episode still leaves >=2 scoring rounds
# under the episode-rejection protocol (results_w1.md).
#
#   bash research/96_selector_foundations/scripts/run_w2.sh
#
# Lane layout: GPU0 = dense arms, GPU1 = llama arms.
set -uo pipefail
cd /data/smcho/self-spec-moe

PHASE=research/96_selector_foundations
LOG="$PHASE/logs"; mkdir -p "$LOG" "$PHASE/data/w2"

export PATH="/data/smcho/self-spec-moe/.venv/bin:$PATH"
export HF_HOME=/data/smcho/huggingface
export VLLM_CACHE_ROOT=/data/smcho/.cache/vllm
export TRITON_CACHE_DIR=/data/smcho/.cache/triton
export TMPDIR=/data/smcho/tmp

COMMON="VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1 VLLM_SELF_SPEC_CPU_ORCH=1"
SHARED="$COMMON VLLM_SELF_SPEC_SHARED_KV=1 VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE=1 VLLM_SELF_SPEC_SKIP_PREFILL_DRAFT=1 VLLM_SELF_SPEC_DRAFT_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1"

SEED="${W2_SEED:-0}"
PIN="${W2_PIN:-0}"
export W2_ITERS="${W2_ITERS:-4}"

cleanup() {
  local gpu="$1"
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$gpu" 2>/dev/null); do
    kill -9 "$p" 2>/dev/null
  done
  local waited=0
  while [ $waited -lt 180 ]; do
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" 2>/dev/null)
    [ "${used:-0}" -le 2000 ] && break
    sleep 10; waited=$((waited + 10))
  done
  sleep 5
}

lane() {      # gpu  arch  cores
  local gpu="$1" arch="$2" cores="$3"
  local wrap=""
  [ "$PIN" = 1 ] && wrap="taskset -c $cores"
  for win in off 512 2048; do
    local env=""
    [ "$win" != off ] && env="$SHARED"
    local out="$PHASE/data/w2/w2_${arch}_w${win}_notune_s${SEED}.json"
    if [ -f "$out" ]; then echo "[W2] skip $arch/w$win (exists)"; continue; fi
    echo "[W2] === gpu=$gpu $arch w=$win seed=$SEED pin=$PIN ==="
    cleanup "$gpu"
    env CUDA_VISIBLE_DEVICES="$gpu" \
        W2_ARCH="$arch" W2_WINDOW="$win" W2_TUNE=0 W2_SEED="$SEED" \
        W2_OUT="$out" \
        $env \
        timeout 5400 $wrap .venv/bin/python \
          "$PHASE/scripts/run_w2_currency.py" \
          >> "$LOG/w2_${arch}.log" 2>&1
    [ $? -ne 0 ] && echo "[W2] FAILED $arch/w$win (see $LOG/w2_${arch}.log)"
  done
  cleanup "$gpu"
  echo "[W2] lane done gpu=$gpu arch=$arch"
}

lane 0 dense 0-15 &
L0=$!
lane 1 llama 16-31 &
L1=$!
wait $L0 $L1
echo "[W2] matrix complete"
ls -la "$PHASE/data/w2/"
