#!/bin/bash
# Discriminator: was the w8int8 "quant failure" in C1 a real limitation
# or the torch-compile cache-key collision? Re-measure the SAME arm on
# the SAME documents (offset 0), changing ONLY VLLM_CACHE_ROOT.
set -uo pipefail
REPO=/data/smcho/self-spec-moe; PHASE=$REPO/research/94_composition
P82DATA=$REPO/research/82_runtime_switching/data
COMPILE=$REPO/research/93_c1_grid/scripts/compile_cells_93.py
export HF_HOME=/data/smcho/huggingface; export PATH="$REPO/.venv/bin:$PATH"; cd "$REPO"
GPU="${AN_GPU:-0}"; ARCH="${AN_ARCH:-dense}"
COMMON="VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1 VLLM_SELF_SPEC_CPU_ORCH=1"
SHARED="$COMMON VLLM_SELF_SPEC_SHARED_KV=1 VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE=1 VLLM_SELF_SPEC_SKIP_PREFILL_DRAFT=1 VLLM_SELF_SPEC_DRAFT_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1"
case "$ARCH" in
  dense) M="Qwen/Qwen3-8B";    DR="/data/smcho/ckpts/Qwen3-8B-W8A16-INT8-sym";;
  llama) M="NousResearch/Meta-Llama-3.1-8B-Instruct"
         DR="/data/smcho/ckpts/Llama31-8B-Instruct-W8A16-INT8-sym";;
esac
gpu_cleanup() {
  local u; u=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$GPU")
  for p in $(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | tr -d ',' | awk -v u="$u" '$2==u{print $1}'); do
    [ "$(ps -o user= -p "$p" 2>/dev/null)" = "smcho" ] || continue
    kill "$p" 2>/dev/null; sleep 2; kill -0 "$p" 2>/dev/null && kill -9 "$p" 2>/dev/null
  done; sleep 3
}
for K in k2 k4; do
  tag="w8probe_${ARCH}_${K}"; csv="anom_${tag}.csv"
  [ -f "$P82DATA/$csv" ] && grep -q "^${K}," "$P82DATA/$csv" && { echo "[W8] skip $tag"; continue; }
  echo "[W8] measure $tag (isolated cache, ORIGINAL documents)"
  env CUDA_VISIBLE_DEVICES=$GPU COMPILE_MODEL="$M" COMPILE_DRAFT="$DR" COMPILE_TP=1 \
      COMPILE_BATCHES="1,8,32" COMPILE_CTXS="2000,8000,14000" COMPILE_KV_LIMIT=260000 \
      COMPILE_CELLS="$csv" COMPILE_TABLE="anom_${tag}.json" COMPILE_DOC_OFFSET=0 \
      VLLM_CACHE_ROOT="/data/smcho/vllm_cache_w8probe/${ARCH}_${K}" \
      $SHARED timeout 3600 .venv/bin/python "$COMPILE" --measure "$K" \
      >> "$PHASE/logs/w8probe_${ARCH}.log" 2>&1 || echo "[W8] FAIL $tag"
  gpu_cleanup
done
cp -f "$P82DATA"/anom_w8probe_*.csv "$PHASE/data/" 2>/dev/null
echo "[W8] DONE $ARCH"
