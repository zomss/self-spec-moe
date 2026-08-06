#!/usr/bin/env bash
# W9: isolate the MoE b32/R2 protocol confound (results_w8.md §6).
#
# The cell arms under W7 (S=1.087) but loses unperturbed under the
# fixed-256 constant-width protocol (S=0.928), with tau unchanged --
# so the difference is cost-side. Two confounds were entangled:
#   (a) natural-EOS batch DRAIN vs constant width
#   (b) max_num_seqs 64 (W7's 1,8,32,64 sweep) vs 32 (single-batch boots)
#
# 2x2 design, all at b32/R2/TP2/notune/K4, S = uncond/off per cell:
#   (natural, 64) = W7            S = 1.087   [banked]
#   (fixed,   32) = W8b + W8d     S = 0.928   [banked]
#   (fixed,   64) = THIS SCRIPT, arms A       -> isolates (b) at fixed width
#   (natural, 32) = THIS SCRIPT, arms B       -> isolates (a) at max_seqs 32
# Comparing the new (fixed,64) against both banked cells attributes each
# effect; (natural,32) closes the square and exposes any interaction.
set -uo pipefail
cd /data/smcho/self-spec-moe
PHASE=research/96_selector_foundations
LOG="$PHASE/logs"; mkdir -p "$LOG" "$PHASE/data/w9"

export PATH="/data/smcho/self-spec-moe/.venv/bin:$PATH"
export HF_HOME=/data/smcho/huggingface VLLM_CACHE_ROOT=/data/smcho/.cache/vllm
export TRITON_CACHE_DIR=/data/smcho/.cache/triton TMPDIR=/data/smcho/tmp
COMMON="VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1 VLLM_SELF_SPEC_CPU_ORCH=1"
SHARED="$COMMON VLLM_SELF_SPEC_SHARED_KV=1 VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE=1 VLLM_SELF_SPEC_SKIP_PREFILL_DRAFT=1 VLLM_SELF_SPEC_DRAFT_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1"
MODEL="Qwen/Qwen3-30B-A3B"; DRAFT="$HOME/ckpts/Qwen3-30B-A3B-W4A16-INT4-sym"

cleanup() {
  for g in 0 1; do
    for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$g" 2>/dev/null); do
      kill -9 "$p" 2>/dev/null; done
  done
  local w=0
  while [ $w -lt 180 ]; do
    local busy=0
    for g in 0 1; do
      local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null)
      [ "${u:-0}" -gt 2000 ] && busy=1
    done
    [ $busy -eq 0 ] && break
    sleep 10; w=$((w + 10))
  done
  sleep 5
}

boot() {  # arm(off|unc) flen maxseqs tag
  local arm="$1" flen="$2" ms="$3" tag="$4"
  local out="$PHASE/data/w9/w9_moe_${tag}.json"
  [ -f "$out" ] && { echo "[W9] skip $tag"; return 0; }
  local draft=off extra=""
  [ "$arm" = "unc" ] && { draft="$DRAFT"; extra="$SHARED"; }
  echo "[W9] === $tag (arm=$arm flen=$flen max_seqs=$ms) ==="
  cleanup
  env CUDA_VISIBLE_DEVICES=0,1 \
      G93_MODEL="$MODEL" G93_TP=2 G93_DRAFT="$draft" G93_K=4 \
      G93_BATCHES=32 G93_DATASETS=R2 G93_ITERS=3 \
      G93_CEILING=16384 G93_MAXLEN=24576 G93_TUNE=0 \
      G93_FIXED_LEN="$flen" G93_MAX_SEQS="$ms" \
      G93_OUT="$out" G93_TAG="w9_moe_$tag" \
      $extra \
      timeout 3600 .venv/bin/python research/93_c1_grid/scripts/run_grid.py \
        >> "$LOG/w9_moe_$tag.log" 2>&1 || echo "[W9] FAILED $tag"
}

# arms A: fixed-256 width, max_num_seqs=64  (isolates the max_seqs effect)
boot off 256 64 fixed256_ms64_off
boot unc 256 64 fixed256_ms64_unc
# arms B: natural EOS, max_num_seqs=32      (isolates the drain effect)
boot off 0   32 natural_ms32_off
boot unc 0   32 natural_ms32_unc
cleanup
echo "[W9] done: $(ls $PHASE/data/w9/*.json 2>/dev/null | wc -l)/4"
