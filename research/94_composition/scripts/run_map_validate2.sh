#!/bin/bash
# Validate the remaining content-EXPOSED C2 map cells: b1 cells whose
# winner contains a 512 window (content-sensitive) and is therefore
# decided by a single document draw.
#   dense  b1/c2000   q-hum_w-512_s-b2 K2    vs q-hum_w-none_s-none K2
#   llama  b1/c14000  q-w4a16_w-512_s-b2 K2  vs q-w4a16_w-none_s-none K2
set -uo pipefail
REPO=/data/smcho/self-spec-moe; PHASE=$REPO/research/94_composition
P82DATA=$REPO/research/82_runtime_switching/data
COMPILE=$REPO/research/93_c1_grid/scripts/compile_cells_93.py
export HF_HOME=/data/smcho/huggingface; export PATH="$REPO/.venv/bin:$PATH"; cd "$REPO"
GPU="${AN_GPU:-0}"; ARCH="${AN_ARCH:-dense}"
HUM="VLLM_DISABLED_KERNELS=MacheteLinearKernel,CutlassW4A8LinearKernel,AllSparkLinearKernel"
COMMON="VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1 VLLM_SELF_SPEC_CPU_ORCH=1"
SHARED="$COMMON VLLM_SELF_SPEC_SHARED_KV=1 VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE=1 VLLM_SELF_SPEC_SKIP_PREFILL_DRAFT=1 VLLM_SELF_SPEC_DRAFT_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1"
W512="VLLM_SELF_SPEC_DRAFT_KV_WINDOW=512 VLLM_SELF_SPEC_DRAFT_KV_SINKS=16 VLLM_SELF_SPEC_DRAFT_FULLCG=1"
case "$ARCH" in
  dense) M="Qwen/Qwen3-8B"; DR="$HOME/ckpts/Qwen3-8B-W4A8-gptq"; QE="$HUM"
         CTX=2000; WIN_EXTRA="$W512 VLLM_SELF_SPEC_DRAFT_SKIP_LAYERS=2,8";;
  llama) M="NousResearch/Meta-Llama-3.1-8B-Instruct"
         DR="$HOME/ckpts/Llama31-8B-Instruct-W4A16-INT4-sym"; QE=""
         CTX=14000; WIN_EXTRA="$W512 VLLM_SELF_SPEC_DRAFT_SKIP_LAYERS=3,8";;
esac
gpu_cleanup() {
  local u; u=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$GPU")
  for p in $(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | tr -d ',' | awk -v u="$u" '$2==u{print $1}'); do
    [ "$(ps -o user= -p "$p" 2>/dev/null)" = "smcho" ] || continue
    kill "$p" 2>/dev/null; sleep 2; kill -0 "$p" 2>/dev/null && kill -9 "$p" 2>/dev/null
  done; sleep 3
}
for OFF in 16 32 48; do
  for ROLE in win sgl; do
    [ "$ROLE" = win ] && EX="$WIN_EXTRA" || EX=""
    tag="map2_${ARCH}_${ROLE}_off${OFF}"; csv="anom_${tag}.csv"
    [ -f "$P82DATA/$csv" ] && grep -q '^k2,' "$P82DATA/$csv" && { echo "[MV2] skip $tag"; continue; }
    echo "[MV2] measure $tag ctx=$CTX"
    env CUDA_VISIBLE_DEVICES=$GPU COMPILE_MODEL="$M" COMPILE_DRAFT="$DR" COMPILE_TP=1 \
        COMPILE_BATCHES="1" COMPILE_CTXS="$CTX" COMPILE_KV_LIMIT=260000 \
        COMPILE_CELLS="$csv" COMPILE_TABLE="anom_${tag}.json" COMPILE_DOC_OFFSET="$OFF" \
        VLLM_CACHE_ROOT="/data/smcho/vllm_cache_anom/${tag}" \
        $SHARED $QE $EX timeout 3600 .venv/bin/python "$COMPILE" --measure k2 \
        >> "$PHASE/logs/map_validate2_${ARCH}.log" 2>&1 || echo "[MV2] FAIL $tag"
    gpu_cleanup
  done
done
cp -f "$P82DATA"/anom_*.csv "$PHASE/data/" 2>/dev/null
echo "[MV2] DONE $ARCH"
