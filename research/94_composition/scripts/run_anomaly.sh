#!/bin/bash
# b1/8k acceptance anomaly: narrow windows (128/512) lose ~20-25% tau at
# ctx 8000 on BOTH arches, but are fine at 2k and recover at 14k.
# Truncation alone cannot be non-monotone, so discriminate:
#   A/B  fine ctx sweep      -> is the dip localised to 8000, or a valley?
#   C/D  same ctx, new docs  -> is it the CONTENT of the 8k prompt pack?
set -uo pipefail
REPO=/data/smcho/self-spec-moe
PHASE=$REPO/research/94_composition
P82DATA=$REPO/research/82_runtime_switching/data
COMPILE=$REPO/research/93_c1_grid/scripts/compile_cells_93.py
export HF_HOME=/data/smcho/huggingface
export PATH="$REPO/.venv/bin:$PATH"
cd "$REPO"
GPU="${AN_GPU:-0}"; MODEL="Qwen/Qwen3-8B"
SWEEP="2000,4000,6000,7000,8000,9000,10000,12000,14000"

COMMON="VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1 VLLM_SELF_SPEC_CPU_ORCH=1"
SHARED="$COMMON VLLM_SELF_SPEC_SHARED_KV=1 VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE=1 VLLM_SELF_SPEC_SKIP_PREFILL_DRAFT=1 VLLM_SELF_SPEC_DRAFT_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1"

gpu_cleanup() {
  local u; u=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$GPU")
  for p in $(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | tr -d ',' | awk -v u="$u" '$2==u{print $1}'); do
    [ "$(ps -o user= -p "$p" 2>/dev/null)" = "smcho" ] || continue
    kill "$p" 2>/dev/null; sleep 2; kill -0 "$p" 2>/dev/null && kill -9 "$p" 2>/dev/null
  done
  sleep 3
}

run() {  # name  window  ctxs  doc_offset
  local name="$1" win="$2" ctxs="$3" off="$4"
  local csv="anom_${name}.csv"
  [ -f "$P82DATA/$csv" ] && grep -q "^k2," "$P82DATA/$csv" && { echo "[ANOM] skip $name"; return; }
  echo "[ANOM] measure $name win=$win ctxs=$ctxs off=$off"
  env CUDA_VISIBLE_DEVICES=$GPU COMPILE_MODEL="$MODEL" COMPILE_DRAFT="$MODEL" \
      COMPILE_TP=1 COMPILE_BATCHES="1" COMPILE_CTXS="$ctxs" \
      COMPILE_KV_LIMIT=260000 COMPILE_CELLS="$csv" \
      COMPILE_TABLE="anom_${name}.json" COMPILE_DOC_OFFSET="$off" \
      VLLM_CACHE_ROOT="/data/smcho/vllm_cache_anom/${name}" \
      VLLM_SELF_SPEC_DRAFT_KV_WINDOW=$win VLLM_SELF_SPEC_DRAFT_KV_SINKS=16 \
      VLLM_SELF_SPEC_DRAFT_FULLCG=1 \
      $SHARED timeout 3600 .venv/bin/python "$COMPILE" --measure k2 \
      >> "$PHASE/logs/anomaly_g${GPU}.log" 2>&1 || echo "[ANOM] FAIL $name"
  gpu_cleanup
}

for job in $AN_JOBS; do
  case "$job" in
    A) run "win512_sweep"   512  "$SWEEP" 0 ;;
    B) run "win8192_sweep" 8192  "$SWEEP" 0 ;;
    C) run "win512_8k_off16" 512 "8000"  16 ;;
    D) run "win512_8k_off32" 512 "8000"  32 ;;
    E) run "win128_sweep"   128  "$SWEEP" 0 ;;
  esac
done
cp -f "$P82DATA"/anom_*.csv "$PHASE/data/" 2>/dev/null
echo "[ANOM] DONE jobs=$AN_JOBS"
