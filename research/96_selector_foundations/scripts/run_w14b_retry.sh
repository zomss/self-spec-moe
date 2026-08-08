#!/usr/bin/env bash
# W14/B retry: re-run only the boots that hit the intermittent draft
# graph-capture wedge. All four failures stopped at the identical point --
# after the draft's "Initial profiling/warmup run", before "Draft FULL-CG"
# and CUDA-graph capture -- which is the documented wedge, not a config
# bug (it struck both compiled and cache-loaded boots, and every
# configuration succeeded at least once).
#
# Timeout cut 5400 -> 1500 s: a healthy boot reaches capture in ~2 min and
# finishes in ~12-15 min, so a 90-min timeout was paying 6x a good boot per
# hang. Up to 3 attempts per missing boot.
set -uo pipefail
cd /data/smcho/self-spec-moe
PHASE=research/96_selector_foundations
LOG="$PHASE/logs/w14"; mkdir -p "$LOG"
export PATH="/data/smcho/self-spec-moe/.venv/bin:$PATH"
export HF_HOME=/data/smcho/huggingface VLLM_CACHE_ROOT=/data/smcho/.cache/vllm
export TRITON_CACHE_DIR=/data/smcho/.cache/triton TMPDIR=/data/smcho/tmp
export VLLM_DISABLED_KERNELS=MacheteLinearKernel,CutlassW4A8LinearKernel,AllSparkLinearKernel
export VLLM_USE_FLASHINFER_SAMPLER=0
export LD_LIBRARY_PATH="/usr/local/cuda-12/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
COMMON="VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1 VLLM_SELF_SPEC_CPU_ORCH=1"
SHARED="$COMMON VLLM_SELF_SPEC_SHARED_KV=1 VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE=1 VLLM_SELF_SPEC_SKIP_PREFILL_DRAFT=1 VLLM_SELF_SPEC_DRAFT_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1"
MODEL="Qwen/Qwen3-8B"; DRAFT="/data/smcho/ckpts/Qwen3-8B-W4A8-gptq"
GPU="${W14_GPU:-1}"

cleanup() {
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$GPU" 2>/dev/null); do
    [ "$(ps -o user= -p "$p" 2>/dev/null)" = "smcho" ] || continue
    kill -9 "$p" 2>/dev/null
  done
  local w=0
  while [ $w -lt 180 ]; do
    local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null)
    [ "${u:-0}" -le 2000 ] && break; sleep 10; w=$((w+10))
  done
  sleep 5
}

# name window
TARGETS=("w512 512 1" "w512 512 3" "woff 0 1" "woff 0 2")
for t in "${TARGETS[@]}"; do
  set -- $t; name=$1; window=$2; n=$3
  tag="w14b_${name}_boot${n}"; out="$PHASE/data/w14/${tag}.json"
  for attempt in 1 2 3; do
    if [ -s "$out" ] && grep -q '"complete": true' "$out"; then break; fi
    echo "[RETRY] $tag attempt $attempt"
    cleanup
    env CUDA_VISIBLE_DEVICES="$GPU" \
        W14_MODEL="$MODEL" W14_DRAFT="$DRAFT" W14_K=4 \
        W14_WINDOW="$window" W14_SKIP="" \
        W14_REGIMES="R5,R5cot" W14_BATCHES="1,8" \
        W14_FIXED_LEN=512 W14_ITERS=4 W14_SEEDS="0,1" \
        W14_REALIZATION="$([ "$window" = 0 ] && echo piecewise-nowindow || echo piecewise)" \
        W14_OUT="$out" W14_TAG="$tag" $SHARED \
        timeout 1500 .venv/bin/python "$PHASE/scripts/w14_measure.py" \
          >> "$LOG/${tag}.log" 2>&1 || echo "[RETRY] wedge on $tag attempt $attempt"
  done
done
cleanup
echo "[RETRY] done: $(ls $PHASE/data/w14/w14b_*.json 2>/dev/null | wc -l)/12"
