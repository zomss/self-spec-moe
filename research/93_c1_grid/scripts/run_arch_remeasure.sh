#!/bin/bash
# Re-measure the remaining collision-implicated Stage-A arms (mla, moe,
# llama) with the cache-key fix in place. Env copied verbatim from
# run_stage_a.sh so the code fix is the only difference.
set -uo pipefail
ARCH=${1:?arch}; ARM=${2:?arm}
REPO=/data/smcho/self-spec-moe; PHASE=$REPO/research/93_c1_grid
P82DATA=$REPO/research/82_runtime_switching/data
COMPILE=$PHASE/scripts/compile_cells_93.py
export HF_HOME=/data/smcho/huggingface; export PATH="$REPO/.venv/bin:$PATH"; cd "$REPO"
GPU="${AR_GPU:-0}"
BATCHES="1,4,8,16,32,64,128"; CTXS="2000,8000,14000"; KS="${AR_KS:-k2 k4 k6}"
COMMON="VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1 VLLM_SELF_SPEC_CPU_ORCH=1"
SHARED="$COMMON VLLM_SELF_SPEC_SHARED_KV=1 VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE=1 VLLM_SELF_SPEC_SKIP_PREFILL_DRAFT=1 VLLM_SELF_SPEC_DRAFT_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1"
case "$ARCH" in
  mla)  MODEL="deepseek-ai/DeepSeek-V2-Lite"; TP=1; KVLIM=250000
        case "$ARM" in
          w4a16) DRAFT="$HOME/ckpts/DeepSeek-V2-Lite-W4A16-INT4-sym"; EXTRA="$SHARED VLLM_DISABLED_KERNELS=MarlinLinearKernel";;
          w8chan) DRAFT="$HOME/ckpts/DeepSeek-V2-Lite-W8A16-INT8-chan"; EXTRA="$SHARED";;
        esac;;
  moe)  MODEL="Qwen/Qwen3-30B-A3B"; TP=2; KVLIM=130000
        case "$ARM" in
          w4a16) DRAFT="$HOME/ckpts/Qwen3-30B-A3B-W4A16-INT4-sym"; EXTRA="$SHARED";;
          w8chan) DRAFT="$HOME/ckpts/Qwen3-30B-A3B-W8A16-INT8-chan"; EXTRA="$SHARED";;
        esac;;
  llama) MODEL="NousResearch/Meta-Llama-3.1-8B-Instruct"; TP=1; KVLIM=260000
        DRAFT="$HOME/ckpts/Llama31-8B-Instruct-W4A16-INT4-sym"; EXTRA="$SHARED";;
esac
gpu_cleanup() {
  for i in ${GPU//,/ }; do
    local u; u=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $i)
    for p in $(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | tr -d ',' | awk -v u="$u" '$2==u{print $1}'); do
      [ "$(ps -o user= -p "$p" 2>/dev/null)" = "smcho" ] || continue
      kill "$p" 2>/dev/null; sleep 2; kill -0 "$p" 2>/dev/null && kill -9 "$p" 2>/dev/null
    done
  done; sleep 3
}
csv="cells_93v2_${ARCH}_${ARM}.csv"
for karm in $KS; do
  if [ -f "$P82DATA/$csv" ] && grep -q "^${karm}," "$P82DATA/$csv"; then echo "[AR:$ARCH/$ARM] skip $karm"; continue; fi
  echo "[AR:$ARCH/$ARM] measure $karm"
  env CUDA_VISIBLE_DEVICES=$GPU COMPILE_MODEL="$MODEL" COMPILE_DRAFT="$DRAFT" \
      COMPILE_TP=$TP COMPILE_BATCHES="$BATCHES" COMPILE_CTXS="$CTXS" \
      COMPILE_KV_LIMIT=$KVLIM COMPILE_CELLS="$csv" \
      COMPILE_TABLE="table_93v2_${ARCH}_${ARM}.json" \
      $EXTRA timeout 5400 .venv/bin/python "$COMPILE" --measure "$karm" \
      >> "$PHASE/logs/arch_remeasure_${ARCH}.log" 2>&1 || echo "[AR:$ARCH/$ARM] FAIL $karm"
  gpu_cleanup
done
cp -f "$P82DATA"/cells_93v2_${ARCH}_*.csv "$PHASE/data/" 2>/dev/null
echo "[AR:$ARCH/$ARM] ARCH-REMEASURE-${ARCH}-${ARM}-DONE"
