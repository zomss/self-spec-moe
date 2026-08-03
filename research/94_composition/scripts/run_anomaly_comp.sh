#!/bin/bash
# Re-measure the COMPOSITION arm that lost at b1/c8000, under re-sampled
# content. The window single recovers +29/+32% on fresh documents; this
# asks whether the composition -- and therefore the C2 winner at that
# cell -- recovers with it.
set -uo pipefail
REPO=/data/smcho/self-spec-moe
PHASE=$REPO/research/94_composition
P82DATA=$REPO/research/82_runtime_switching/data
COMPILE=$REPO/research/93_c1_grid/scripts/compile_cells_93.py
export HF_HOME=/data/smcho/huggingface
export PATH="$REPO/.venv/bin:$PATH"
cd "$REPO"
GPU="${AN_GPU:-0}"; ARCH="${AN_ARCH:-dense}"
HUM="VLLM_DISABLED_KERNELS=MacheteLinearKernel,CutlassW4A8LinearKernel,AllSparkLinearKernel"
case "$ARCH" in
  dense) MODEL="Qwen/Qwen3-8B"; DRAFT="$HOME/ckpts/Qwen3-8B-W4A8-gptq"; QE="$HUM";;
  llama) MODEL="NousResearch/Meta-Llama-3.1-8B-Instruct"
         DRAFT="$HOME/ckpts/Llama31-8B-Instruct-W4A16-INT4-sym"; QE="";;
esac
COMMON="VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1 VLLM_SELF_SPEC_CPU_ORCH=1"
SHARED="$COMMON VLLM_SELF_SPEC_SHARED_KV=1 VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE=1 VLLM_SELF_SPEC_SKIP_PREFILL_DRAFT=1 VLLM_SELF_SPEC_DRAFT_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1"
gpu_cleanup() {
  local u; u=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$GPU")
  for p in $(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | tr -d ',' | awk -v u="$u" '$2==u{print $1}'); do
    [ "$(ps -o user= -p "$p" 2>/dev/null)" = "smcho" ] || continue
    kill "$p" 2>/dev/null; sleep 2; kill -0 "$p" 2>/dev/null && kill -9 "$p" 2>/dev/null
  done; sleep 3
}
for OFF in ${AN_OFFSETS:-16 32}; do
  for WIN in ${AN_WINS:-512 0}; do   # WIN=0 -> the quant-only SINGLE, same documents (control)
    tag="comp_${ARCH}_off${OFF}_w${WIN}"
    csv="anom_${tag}.csv"
    [ -f "$P82DATA/$csv" ] && grep -q '^k2,' "$P82DATA/$csv" && { echo "[ANOMC] skip $tag"; continue; }
    W=""; [ "$WIN" != "0" ] && W="VLLM_SELF_SPEC_DRAFT_KV_WINDOW=$WIN VLLM_SELF_SPEC_DRAFT_KV_SINKS=16 VLLM_SELF_SPEC_DRAFT_FULLCG=1"
    echo "[ANOMC] measure $tag"
    env CUDA_VISIBLE_DEVICES=$GPU COMPILE_MODEL="$MODEL" COMPILE_DRAFT="$DRAFT" \
        COMPILE_TP=1 COMPILE_BATCHES="1" COMPILE_CTXS="8000" \
        COMPILE_KV_LIMIT=260000 COMPILE_CELLS="$csv" COMPILE_TABLE="anom_${tag}.json" \
        COMPILE_DOC_OFFSET="$OFF" VLLM_CACHE_ROOT="/data/smcho/vllm_cache_anom/${tag}" \
        $SHARED $QE $W timeout 3600 .venv/bin/python "$COMPILE" --measure k2 \
        >> "$PHASE/logs/anomaly_comp_${ARCH}.log" 2>&1 || echo "[ANOMC] FAIL $tag"
    gpu_cleanup
  done
done
cp -f "$P82DATA"/anom_*.csv "$PHASE/data/" 2>/dev/null
echo "[ANOMC] DONE $ARCH"
