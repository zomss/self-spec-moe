#!/usr/bin/env bash
# W14 item B — matched-cell cost-transfer pilot (the first GPU gate).
#
# Question: does q_a(x) transfer between content regimes when x and the
# amount of work are actually matched?
#
# Matrix: Qwen3-8B dense; regimes R5, R5cot; FIRST 512 generated tokens
# for BOTH (matched suffix interval); batches 1, 8; configurations
# s-none/w512, s-none/w2048, s-none/w-off; K=4.
#
# 12 boots nominal: 3 configurations x 3 boots + 3 AR anchor boots.
# THREE boots (not two) because the resampling unit is the boot: n=2
# leaves one dof for a 95% interval, and a source-B episode shifts a
# whole boot ~11% (W1) which at n=2 is undetectable -- a uniformly
# shifted boot reads as a transfer FAILURE rather than a rejection.
#
# Boot order is randomized (fixed pre-declared permutation, seed 20260808)
# so configuration and drift are not confounded.
set -uo pipefail
cd /data/smcho/self-spec-moe
PHASE=research/96_selector_foundations
LOG="$PHASE/logs/w14"; mkdir -p "$LOG" "$PHASE/data/w14"

export PATH="/data/smcho/self-spec-moe/.venv/bin:$PATH"
export HF_HOME=/data/smcho/huggingface VLLM_CACHE_ROOT=/data/smcho/.cache/vllm
export TRITON_CACHE_DIR=/data/smcho/.cache/triton TMPDIR=/data/smcho/tmp
export VLLM_DISABLED_KERNELS=MacheteLinearKernel,CutlassW4A8LinearKernel,AllSparkLinearKernel
# W13 environment fixes (2026-08-07 home reset) -- see run_w13_taustar.sh
export VLLM_USE_FLASHINFER_SAMPLER=0
export LD_LIBRARY_PATH="/usr/local/cuda-12/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

COMMON="VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1 VLLM_SELF_SPEC_CPU_ORCH=1"
SHARED="$COMMON VLLM_SELF_SPEC_SHARED_KV=1 VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE=1 VLLM_SELF_SPEC_SKIP_PREFILL_DRAFT=1 VLLM_SELF_SPEC_DRAFT_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1"

MODEL="Qwen/Qwen3-8B"
DRAFT="/data/smcho/ckpts/Qwen3-8B-W4A8-gptq"
GPU="${W14_GPU:-1}"

# Pre-declared randomized boot order (permutation frozen before the run).
# name|draft|window|realization
ORDER=(
  "w2048|spec|2048|piecewise"
  "ar|off|0|ar"
  "woff|spec|0|piecewise-nowindow"
  "w512|spec|512|piecewise"
  "ar|off|0|ar"
  "w512|spec|512|piecewise"
  "w2048|spec|2048|piecewise"
  "woff|spec|0|piecewise-nowindow"
  "w512|spec|512|piecewise"
  "ar|off|0|ar"
  "woff|spec|0|piecewise-nowindow"
  "w2048|spec|2048|piecewise"
)

cleanup() {
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$GPU" 2>/dev/null); do
    [ "$(ps -o user= -p "$p" 2>/dev/null)" = "smcho" ] || continue
    kill -9 "$p" 2>/dev/null
  done
  local w=0
  while [ $w -lt 180 ]; do
    local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null)
    [ "${u:-0}" -le 2000 ] && break; sleep 10; w=$((w + 10))
  done
  sleep 5
}

declare -A SEEN
i=0
for spec in "${ORDER[@]}"; do
  i=$((i + 1))
  IFS='|' read -r name arm window realization <<< "$spec"
  n=$(( ${SEEN[$name]:-0} + 1 )); SEEN[$name]=$n
  tag="w14b_${name}_boot${n}"
  out="$PHASE/data/w14/${tag}.json"
  if [ -s "$out" ] && grep -q '"complete": true' "$out"; then
    echo "[W14B] skip $tag (done)"; continue
  fi
  echo "[W14B] === ($i/12) $tag on gpu $GPU ==="
  cleanup
  d="off"; extra=""
  [ "$arm" = "spec" ] && { d="$DRAFT"; extra="$SHARED"; }
  env CUDA_VISIBLE_DEVICES="$GPU" \
      W14_MODEL="$MODEL" W14_DRAFT="$d" W14_K=4 \
      W14_WINDOW="$window" W14_SKIP="" \
      W14_REGIMES="R5,R5cot" W14_BATCHES="1,8" \
      W14_FIXED_LEN=512 W14_ITERS=4 W14_SEEDS="0,1" \
      W14_REALIZATION="$realization" \
      W14_OUT="$out" W14_TAG="$tag" \
      $extra \
      timeout 5400 .venv/bin/python "$PHASE/scripts/w14_measure.py" \
        >> "$LOG/${tag}.log" 2>&1 \
        || echo "[W14B] FAILED $tag (continuing)"
done
cleanup
echo "[W14B] done: $(ls $PHASE/data/w14/w14b_*.json 2>/dev/null | wc -l)/12"
