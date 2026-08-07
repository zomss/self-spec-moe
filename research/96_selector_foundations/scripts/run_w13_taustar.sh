#!/usr/bin/env bash
# W13: does tau* reproduce the dense selector? (w13_tau_star_validation.md)
# 11 boots on Qwen3-8B, TP1, two GPU lanes:
#   1 off (gives T) + 5 profiled uncond (D,V,C) + 5 unprofiled uncond (tau, S)
# Profiled and unprofiled are separate boots by design: profiled serving
# numbers are discarded (pre-registered since W8), so actual S needs an
# unperturbed run.
set -uo pipefail
cd /data/smcho/self-spec-moe
PHASE=research/96_selector_foundations
LOG="$PHASE/logs"; mkdir -p "$LOG" "$PHASE/data/w13"

export PATH="/data/smcho/self-spec-moe/.venv/bin:$PATH"
export HF_HOME=/data/smcho/huggingface VLLM_CACHE_ROOT=/data/smcho/.cache/vllm
export TRITON_CACHE_DIR=/data/smcho/.cache/triton TMPDIR=/data/smcho/tmp
export VLLM_DISABLED_KERNELS=MacheteLinearKernel,CutlassW4A8LinearKernel,AllSparkLinearKernel
# 2026-08-07: /home/smcho was reset -- the flashinfer JIT cache rebuilt against
# CUDA 13 (libcudart.so.13) inside a torch-cu129 stack and fails to load. All W13
# regimes are temperature 0 (greedy), so the top-k/top-p sampler backend cannot
# affect outputs; disable it rather than mixing CUDA runtimes.
export VLLM_USE_FLASHINFER_SAMPLER=0

COMMON="VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1 VLLM_SELF_SPEC_CPU_ORCH=1"
SHARED="$COMMON VLLM_SELF_SPEC_SHARED_KV=1 VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE=1 VLLM_SELF_SPEC_SKIP_PREFILL_DRAFT=1 VLLM_SELF_SPEC_DRAFT_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1"

MODEL="Qwen/Qwen3-8B"; DRAFT="/data/smcho/ckpts/Qwen3-8B-W4A8-gptq"
FLEN="R1:1024,R5:512,R5cot:3072"

# name|skip|window   (window 0 = full KV)
COMPS=(
  "s-none_w512|none|512"
  "s-none_w2048|none|2048"
  "s-none_woff|none|0"
  "s-2_8_w512|2,8|512"
  "s-2_8_w2048|2,8|2048"
)

cleanup() {
  local g="$1"
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$g" 2>/dev/null); do
    [ "$(ps -o user= -p "$p" 2>/dev/null)" = "smcho" ] || continue
    kill -9 "$p" 2>/dev/null
  done
  local w=0
  while [ $w -lt 180 ]; do
    local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null)
    [ "${u:-0}" -le 2000 ] && break; sleep 10; w=$((w + 10))
  done
  sleep 5
}

boot() {  # gpu tag draft skip window profile_dir_or_empty out
  local gpu="$1" tag="$2" draft="$3" skip="$4" window="$5" pdir="$6" out="$7"
  if [ -s "$out" ] && grep -q '"complete": true' "$out"; then
    echo "[W13] skip $tag (done)"; return 0
  fi
  echo "[W13] === $tag on gpu $gpu ==="
  cleanup "$gpu"
  local penv=""
  if [ -n "$pdir" ]; then
    mkdir -p "$pdir"
    penv="VLLM_SELF_SPEC_PROFILE=1 VLLM_SELF_SPEC_PROFILE_OUT=$pdir"
  fi
  local senv=""
  [ "$draft" != "off" ] && senv="$SHARED"
  env CUDA_VISIBLE_DEVICES="$gpu" \
      G93_MODEL="$MODEL" G93_TP=1 G93_DRAFT="$draft" G93_K=4 \
      G93_WINDOW="$window" G93_SKIP="$skip" \
      G93_BATCHES=1,8 G93_DATASETS=R1,R5,R5cot G93_ITERS=4 \
      G93_CEILING=16384 G93_MAXLEN=20480 \
      G93_TUNE=0 G93_FIXED_LEN="$FLEN" \
      G93_OUT="$out" G93_TAG="w13_$tag" \
      $penv $senv \
      timeout 5400 .venv/bin/python research/93_c1_grid/scripts/run_grid.py \
        >> "$LOG/w13_$tag.log" 2>&1 \
        || echo "[W13] FAILED $tag (continuing)"
}

# 2026-08-07: co-tenant (user jhchoi) took GPU0 with ~14 GiB, so the two-lane
# plan is off. All 11 boots run SEQUENTIALLY on GPU1 at the unchanged
# gpu_memory_utilization=0.90 -- lowering it to squeeze onto GPU0 would resize
# the KV cache and risk preemption at b8/14k, a confound W6's numbers do not
# carry. GPU1 is also the preferred lane per the episode dossier (every
# suppressed cell in this arc sat on GPU0). cleanup() only kills smcho-owned
# processes, so the co-tenant is never touched.
GPU="${W13_GPU:-1}"

run_all() {
  boot "$GPU" off off "" 0 "" "$PHASE/data/w13/w13_off.json"
  for i in 0 1 2 3 4; do
    IFS='|' read -r n sk win <<< "${COMPS[$i]}"
    [ "$sk" = "none" ] && sk=""
    boot "$GPU" "prof_$n" "$DRAFT" "$sk" "$win" \
         "$PHASE/data/w13/prof_$n" "$PHASE/data/w13/prof_$n/serving.json"
    boot "$GPU" "unc_$n" "$DRAFT" "$sk" "$win" "" \
         "$PHASE/data/w13/w13_unc_$n.json"
  done
  cleanup "$GPU"
}

run_all
echo "[W13] done: $(ls $PHASE/data/w13/w13_*.json 2>/dev/null | wc -l) serving, $(ls -d $PHASE/data/w13/prof_* 2>/dev/null | wc -l) profiled"
