#!/bin/bash
# Validate the C2 winner-map cell b1/c8000 under re-sampled content.
# The winner there uses win2048 (content-insensitive) so the SELECTION
# cannot move, but llama's margin is only +4.9% and win2048 sits at 5%
# b1 spread -- so re-measure the winner AND the best single on fresh
# documents and check the margin survives.
set -uo pipefail
REPO=/data/smcho/self-spec-moe; PHASE=$REPO/research/94_composition
P82DATA=$REPO/research/82_runtime_switching/data
COMPILE=$REPO/research/93_c1_grid/scripts/compile_cells_93.py
export HF_HOME=/data/smcho/huggingface; export PATH="$REPO/.venv/bin:$PATH"; cd "$REPO"
GPU="${AN_GPU:-0}"; ARCH="${AN_ARCH:-dense}"
HUM="VLLM_DISABLED_KERNELS=MacheteLinearKernel,CutlassW4A8LinearKernel,AllSparkLinearKernel"
COMMON="VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1 VLLM_SELF_SPEC_CPU_ORCH=1"
SHARED="$COMMON VLLM_SELF_SPEC_SHARED_KV=1 VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE=1 VLLM_SELF_SPEC_SKIP_PREFILL_DRAFT=1 VLLM_SELF_SPEC_DRAFT_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1"
WIN2048="VLLM_SELF_SPEC_DRAFT_KV_WINDOW=2048 VLLM_SELF_SPEC_DRAFT_KV_SINKS=16 VLLM_SELF_SPEC_DRAFT_FULLCG=1"
case "$ARCH" in
  dense) M="Qwen/Qwen3-8B"; DR="$HOME/ckpts/Qwen3-8B-W4A8-gptq"; QE="$HUM"
         KARM=k4
         WIN_EXTRA="$WIN2048 VLLM_SELF_SPEC_DRAFT_SKIP_LAYERS=2,8"   # q-hum_w-2048_s-b2
         SGL_EXTRA=""                                                 # q-hum_w-none_s-none
         ;;
  llama) M="NousResearch/Meta-Llama-3.1-8B-Instruct"
         DR="$HOME/ckpts/Llama31-8B-Instruct-W4A16-INT4-sym"; QE=""
         KARM=k2
         WIN_EXTRA="$WIN2048"                                         # q-w4a16_w-2048_s-none
         SGL_EXTRA=""                                                 # q-w4a16_w-none_s-none
         ;;
esac
gpu_cleanup() {
  local u; u=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$GPU")
  for p in $(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | tr -d ',' | awk -v u="$u" '$2==u{print $1}'); do
    [ "$(ps -o user= -p "$p" 2>/dev/null)" = "smcho" ] || continue
    kill "$p" 2>/dev/null; sleep 2; kill -0 "$p" 2>/dev/null && kill -9 "$p" 2>/dev/null
  done; sleep 3
}
for OFF in ${MV_OFFSETS:-16 32 48}; do
  for ROLE in win sgl; do
    [ "$ROLE" = win ] && EX="$WIN_EXTRA" || EX="$SGL_EXTRA"
    tag="map_${ARCH}_${ROLE}_off${OFF}"; csv="anom_${tag}.csv"
    [ -f "$P82DATA/$csv" ] && grep -q "^${KARM}," "$P82DATA/$csv" && { echo "[MV] skip $tag"; continue; }
    echo "[MV] measure $tag ($KARM)"
    env CUDA_VISIBLE_DEVICES=$GPU COMPILE_MODEL="$M" COMPILE_DRAFT="$DR" COMPILE_TP=1 \
        COMPILE_BATCHES="1" COMPILE_CTXS="8000" COMPILE_KV_LIMIT=260000 \
        COMPILE_CELLS="$csv" COMPILE_TABLE="anom_${tag}.json" COMPILE_DOC_OFFSET="$OFF" \
        VLLM_CACHE_ROOT="/data/smcho/vllm_cache_anom/${tag}" \
        $SHARED $QE $EX timeout 3600 .venv/bin/python "$COMPILE" --measure "$KARM" \
        >> "$PHASE/logs/map_validate_${ARCH}.log" 2>&1 || echo "[MV] FAIL $tag"
    gpu_cleanup
  done
done
cp -f "$P82DATA"/anom_*.csv "$PHASE/data/" 2>/dev/null
echo "[MV] DONE $ARCH"
