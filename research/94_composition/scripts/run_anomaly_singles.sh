#!/bin/bash
# Close the remaining 3 exposed C1 cells (narrow-window singles at
# b1/c8000) with 2 extra document draws each, so the C1 winner map at
# that cell rests on a mean rather than one draw.
set -uo pipefail
REPO=/data/smcho/self-spec-moe; PHASE=$REPO/research/94_composition
P82DATA=$REPO/research/82_runtime_switching/data
COMPILE=$REPO/research/93_c1_grid/scripts/compile_cells_93.py
export HF_HOME=/data/smcho/huggingface; export PATH="$REPO/.venv/bin:$PATH"; cd "$REPO"
GPU="${AN_GPU:-0}"
COMMON="VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1 VLLM_SELF_SPEC_CPU_ORCH=1"
SHARED="$COMMON VLLM_SELF_SPEC_SHARED_KV=1 VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE=1 VLLM_SELF_SPEC_SKIP_PREFILL_DRAFT=1 VLLM_SELF_SPEC_DRAFT_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1"
gpu_cleanup() {
  local u; u=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$GPU")
  for p in $(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | tr -d ',' | awk -v u="$u" '$2==u{print $1}'); do
    [ "$(ps -o user= -p "$p" 2>/dev/null)" = "smcho" ] || continue
    kill "$p" 2>/dev/null; sleep 2; kill -0 "$p" 2>/dev/null && kill -9 "$p" 2>/dev/null
  done; sleep 3
}
for spec in $AN_SPECS; do
  IFS=: read -r arch win <<< "$spec"
  case "$arch" in
    dense) M="Qwen/Qwen3-8B";;
    llama) M="NousResearch/Meta-Llama-3.1-8B-Instruct";;
  esac
  for OFF in 16 32; do
    tag="single_${arch}_win${win}_off${OFF}"; csv="anom_${tag}.csv"
    [ -f "$P82DATA/$csv" ] && grep -q '^k2,' "$P82DATA/$csv" && { echo "[ANOMS] skip $tag"; continue; }
    echo "[ANOMS] measure $tag"
    env CUDA_VISIBLE_DEVICES=$GPU COMPILE_MODEL="$M" COMPILE_DRAFT="$M" COMPILE_TP=1 \
        COMPILE_BATCHES="1" COMPILE_CTXS="8000" COMPILE_KV_LIMIT=260000 \
        COMPILE_CELLS="$csv" COMPILE_TABLE="anom_${tag}.json" COMPILE_DOC_OFFSET="$OFF" \
        VLLM_CACHE_ROOT="/data/smcho/vllm_cache_anom/${tag}" \
        VLLM_SELF_SPEC_DRAFT_KV_WINDOW=$win VLLM_SELF_SPEC_DRAFT_KV_SINKS=16 \
        VLLM_SELF_SPEC_DRAFT_FULLCG=1 \
        $SHARED timeout 3600 .venv/bin/python "$COMPILE" --measure k2 \
        >> "$PHASE/logs/anomaly_singles.log" 2>&1 || echo "[ANOMS] FAIL $tag"
    gpu_cleanup
  done
done
cp -f "$P82DATA"/anom_*.csv "$PHASE/data/" 2>/dev/null
echo "[ANOMS] DONE $AN_SPECS"
