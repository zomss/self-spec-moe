#!/bin/bash
# Cache-key collision A/B. Runs TWO quant drafts that differ ONLY in the
# draft checkpoint, back to back, into ONE SHARED VLLM_CACHE_ROOT:
#   1. w4a16  (primes the shared cache)
#   2. w8int8 (collides pre-fix -> loads w4a16's draft_model graph)
# Pre-fix expectation: w8int8 tau ~1.04. Post-fix: ~2.97.
set -uo pipefail
REPO=/data/smcho/self-spec-moe; PHASE=$REPO/research/94_composition
P82DATA=$REPO/research/82_runtime_switching/data
COMPILE=$REPO/research/93_c1_grid/scripts/compile_cells_93.py
export HF_HOME=/data/smcho/huggingface; export PATH="$REPO/.venv/bin:$PATH"; cd "$REPO"
GPU="${CK_GPU:-0}"; TAG="${CK_TAG:?set CK_TAG=prefix|postfix}"
ROOT="/data/smcho/vllm_cache_ckverify_${TAG}"
rm -rf "$ROOT"    # a genuinely shared, genuinely empty starting cache
MODEL="Qwen/Qwen3-8B"
COMMON="VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1 VLLM_SELF_SPEC_CPU_ORCH=1"
SHARED="$COMMON VLLM_SELF_SPEC_SHARED_KV=1 VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE=1 VLLM_SELF_SPEC_SKIP_PREFILL_DRAFT=1 VLLM_SELF_SPEC_DRAFT_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1"
gpu_cleanup() {
  local u; u=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$GPU")
  for p in $(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | tr -d ',' | awk -v u="$u" '$2==u{print $1}'); do
    [ "$(ps -o user= -p "$p" 2>/dev/null)" = "smcho" ] || continue
    kill "$p" 2>/dev/null; sleep 2; kill -0 "$p" 2>/dev/null && kill -9 "$p" 2>/dev/null
  done; sleep 3
}
run() {  # <name> <ckpt>
  local name=$1 ck=$2
  local csv="anom_ck_${TAG}_${name}.csv"
  rm -f "$P82DATA/$csv"
  echo "[CK:$TAG] measure $name (shared root $ROOT)"
  env CUDA_VISIBLE_DEVICES=$GPU COMPILE_MODEL="$MODEL" COMPILE_DRAFT="$ck" \
      COMPILE_TP=1 COMPILE_BATCHES="1" COMPILE_CTXS="2000" COMPILE_KV_LIMIT=260000 \
      COMPILE_CELLS="$csv" COMPILE_TABLE="anom_ck_${TAG}_${name}.json" \
      VLLM_CACHE_ROOT="$ROOT" \
      $SHARED timeout 3600 .venv/bin/python "$COMPILE" --measure k2 \
      >> "$PHASE/logs/cachekey_${TAG}.log" 2>&1 || echo "[CK:$TAG] FAIL $name"
  gpu_cleanup
}
run w4a16  "$HOME/ckpts/Qwen3-8B-W4A16-INT4"
run w8int8 "$HOME/ckpts/Qwen3-8B-W8A16-INT8-sym"
echo "[CK:$TAG] keys used:"
grep -ohE "torch_compile_cache/[0-9a-f]{10}/rank_0_0/draft_model" "$PHASE/logs/cachekey_${TAG}.log" | sort -u
cp -f "$P82DATA"/anom_ck_*.csv "$PHASE/data/" 2>/dev/null
echo "[CK:$TAG] CACHEKEY-VERIFY-DONE"
