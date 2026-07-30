#!/bin/bash
# Dense retry-ledger backfill on GPU 7 (runs while the main dense
# driver holds GPU 6). Window arms at b<=64: the scratchpad IMA is
# boot-shape-dependent (b128/capture-512; E6-validated shape is b64).
# b128 window cells recorded as realization-unavailable (fast stack).
set -uo pipefail
REPO=/data/smcho/self-spec-moe
PHASE=$REPO/research/93_c1_grid
P82DATA=$REPO/research/82_runtime_switching/data
export HF_HOME=/data/smcho/huggingface
export PATH="$REPO/.venv/bin:$PATH"
cd "$REPO"
GPU=6
MODEL="Qwen/Qwen3-8B"

COMMON="VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1 VLLM_SELF_SPEC_CPU_ORCH=1"
SHARED="$COMMON VLLM_SELF_SPEC_SHARED_KV=1 VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE=1 VLLM_SELF_SPEC_SKIP_PREFILL_DRAFT=1 VLLM_SELF_SPEC_DRAFT_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1"
winstack() { echo "$SHARED VLLM_SELF_SPEC_DRAFT_KV_WINDOW=$1 VLLM_SELF_SPEC_DRAFT_KV_SINKS=16 VLLM_SELF_SPEC_DRAFT_FULLCG=1 VLLM_SELF_SPEC_DRAFT_STEP0_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_WHOLECHAIN=1"; }

gpu_cleanup() {
  local u; u=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $GPU)
  for p in $(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | tr -d ',' | awk -v u="$u" '$2==u{print $1}'); do
    [ "$(ps -o user= -p "$p" 2>/dev/null)" = "smcho" ] || continue
    kill "$p" 2>/dev/null; sleep 2; kill -0 "$p" 2>/dev/null && kill -9 "$p" 2>/dev/null
  done
  sleep 3
}

run_one() {  # <lever> <karm> <draft> <batches> <extra>
  local lever=$1 karm=$2 draft=$3 batches=$4 extra=$5
  local csv="cells_93_dense_${lever}.csv"
  if [ -f "$P82DATA/$csv" ] && grep -q "^${karm}," "$P82DATA/$csv"; then
    echo "[BF] skip $lever/$karm"; return 0
  fi
  echo "[BF] measure $lever/$karm (b<=${batches##*,})"
  env CUDA_VISIBLE_DEVICES=$GPU COMPILE_MODEL="$MODEL" \
      COMPILE_DRAFT="$draft" COMPILE_TP=1 \
      COMPILE_BATCHES="$batches" COMPILE_CTXS="2000,8000,14000" \
      COMPILE_KV_LIMIT=260000 COMPILE_CELLS="$csv" \
      COMPILE_TABLE="table_93_dense_${lever}.json" \
      $extra timeout 3600 .venv/bin/python \
      "$PHASE/scripts/compile_cells_93.py" --measure "$karm" \
      >> "$PHASE/logs/backfill_dense6.log" 2>&1 \
      || echo "[BF] FAIL $lever/$karm"
  gpu_cleanup
}

B64="1,4,8,16,32,64"
BFULL="1,4,8,16,32,64,128"

for k in k2 k4 k6; do run_one win8192 $k "$MODEL" "$B64" "$(winstack 8192)"; done
for k in k2 k4 k6; do run_one fp8dyn $k "$HOME/ckpts/Qwen3-8B-FP8-dynamic" "$BFULL" "$SHARED"; done
run_one w4a8hum k2 "$HOME/ckpts/Qwen3-8B-W4A8-gptq" "$BFULL" "$SHARED VLLM_DISABLED_KERNELS=MacheteLinearKernel,CutlassW4A8LinearKernel,AllSparkLinearKernel"
run_one w4a8hum k2 "$HOME/ckpts/Qwen3-8B-W4A8-gptq" "$BFULL" "$SHARED VLLM_DISABLED_KERNELS=MacheteLinearKernel,CutlassW4A8LinearKernel,AllSparkLinearKernel"

cp -f "$P82DATA"/cells_93_dense_*.csv "$PHASE/data/" 2>/dev/null
echo "[BF] DENSE-BACKFILL6-DONE"

