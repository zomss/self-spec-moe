#!/bin/bash
# Re-measure the C1 w8int8 column with ISOLATED compile caches.
#
# Identical to run_stage_a.sh's w8int8 arm in every respect -- same
# BATCHES/CTXS/KV_LIMIT, same SHARED production bundle, same checkpoint,
# same K set -- except VLLM_CACHE_ROOT is per (arch,K) instead of the
# shared default. The oracle replication showed all 12 llama w8int8 arms
# jump tau 1.8 -> 4.9 when caches are isolated, which is the torch-compile
# cache-key collision, not a property of 8-bit drafting.
# Writes cells_93v2_* so the original column is preserved for comparison.
set -uo pipefail
ARCH=${1:?usage: run_w8int8_remeasure.sh dense|llama}
REPO=/data/smcho/self-spec-moe
PHASE=$REPO/research/93_c1_grid
P82DATA=$REPO/research/82_runtime_switching/data
COMPILE=$PHASE/scripts/compile_cells_93.py
export HF_HOME=/data/smcho/huggingface
export PATH="$REPO/.venv/bin:$PATH"
cd "$REPO"
GPU="${W8_GPU:-0}"
BATCHES="1,4,8,16,32,64,128"; CTXS="2000,8000,14000"; KS="k2 k4 k6"
COMMON="VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1 VLLM_SELF_SPEC_CPU_ORCH=1"
SHARED="$COMMON VLLM_SELF_SPEC_SHARED_KV=1 VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE=1 VLLM_SELF_SPEC_SKIP_PREFILL_DRAFT=1 VLLM_SELF_SPEC_DRAFT_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1"
case "$ARCH" in
  dense) MODEL="Qwen/Qwen3-8B"; KVLIM=260000
         DRAFT="/data/smcho/ckpts/Qwen3-8B-W8A16-INT8-sym";;
  llama) MODEL="NousResearch/Meta-Llama-3.1-8B-Instruct"; KVLIM=260000
         DRAFT="/data/smcho/ckpts/Llama31-8B-Instruct-W8A16-INT8-sym";;
esac
gpu_cleanup() {
  local u; u=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$GPU")
  for p in $(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | tr -d ',' | awk -v u="$u" '$2==u{print $1}'); do
    [ "$(ps -o user= -p "$p" 2>/dev/null)" = "smcho" ] || continue
    kill "$p" 2>/dev/null; sleep 2; kill -0 "$p" 2>/dev/null && kill -9 "$p" 2>/dev/null
  done; sleep 3
}
csv="cells_93v2_${ARCH}_w8int8.csv"
for karm in $KS; do
  if [ -f "$P82DATA/$csv" ] && grep -q "^${karm}," "$P82DATA/$csv"; then
    echo "[W8v2:$ARCH] skip $karm"; continue
  fi
  echo "[W8v2:$ARCH] measure $karm (isolated cache)"
  env CUDA_VISIBLE_DEVICES=$GPU COMPILE_MODEL="$MODEL" COMPILE_DRAFT="$DRAFT" \
      COMPILE_TP=1 COMPILE_BATCHES="$BATCHES" COMPILE_CTXS="$CTXS" \
      COMPILE_KV_LIMIT=$KVLIM COMPILE_CELLS="$csv" \
      COMPILE_TABLE="table_93v2_${ARCH}_w8int8.json" \
      VLLM_CACHE_ROOT="/data/smcho/vllm_cache_w8v2/${ARCH}_${karm}" \
      $SHARED timeout 5400 .venv/bin/python "$COMPILE" --measure "$karm" \
      >> "$PHASE/logs/w8int8_remeasure_${ARCH}.log" 2>&1 \
      || echo "[W8v2:$ARCH] FAIL $karm"
  gpu_cleanup
done
cp -f "$P82DATA"/cells_93v2_${ARCH}_*.csv "$PHASE/data/" 2>/dev/null
echo "[W8v2:$ARCH] W8INT8-REMEASURE-${ARCH}-DONE"
