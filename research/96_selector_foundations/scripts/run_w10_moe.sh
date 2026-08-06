#!/usr/bin/env bash
# W10: MoE Stage-B surface re-measured under EQUAL WORK
# (w10_moe_remeasure.md). 5 arms x 33 cells, TP2 GPUs 0+1, sequential.
# Arms replicate 93/scripts/run_stage_b.sh exactly; the only changes are
# G93_TUNE=0 (notune) and per-regime G93_FIXED_LEN, plus ITERS=4.
set -uo pipefail
cd /data/smcho/self-spec-moe
PHASE=research/96_selector_foundations
LOG="$PHASE/logs"; mkdir -p "$LOG" "$PHASE/data/w10"

export PATH="/data/smcho/self-spec-moe/.venv/bin:$PATH"
export HF_HOME=/data/smcho/huggingface VLLM_CACHE_ROOT=/data/smcho/.cache/vllm
export TRITON_CACHE_DIR=/data/smcho/.cache/triton TMPDIR=/data/smcho/tmp

COMMON="VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1 VLLM_SELF_SPEC_CPU_ORCH=1"
SHARED="$COMMON VLLM_SELF_SPEC_SHARED_KV=1 VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE=1 VLLM_SELF_SPEC_SKIP_PREFILL_DRAFT=1 VLLM_SELF_SPEC_DRAFT_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1"
winplain() { echo "$SHARED VLLM_SELF_SPEC_DRAFT_KV_WINDOW=$1 VLLM_SELF_SPEC_DRAFT_KV_SINKS=16"; }

MODEL="Qwen/Qwen3-30B-A3B"
CKPT="$HOME/ckpts/Qwen3-30B-A3B-W4A16-INT4-sym"
# per-regime natural p50 from the Stage-B off arm (median across batches)
FLEN="R1:280,R2:205,R3:265,R4:815,R5:330,R5cot:1030,R6:76,R7:23,R8:1250"

ARMLIST=(
  "off|off|0|0|"
  "w4a16_k2|$CKPT|2|0|$SHARED VLLM_DISABLED_KERNELS=MacheteLinearKernel"
  "w4a16_k3|$CKPT|3|0|$SHARED VLLM_DISABLED_KERNELS=MacheteLinearKernel"
  "win2048_k3|self|3|2048|$(winplain 2048)"
  "win8192_k3|self|3|8192|$(winplain 8192)"
)

cleanup() {
  for g in 0 1; do
    for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$g" 2>/dev/null); do
      [ "$(ps -o user= -p "$p" 2>/dev/null)" = "smcho" ] || continue
      kill -9 "$p" 2>/dev/null
    done
  done
  local w=0
  while [ $w -lt 180 ]; do
    local busy=0
    for g in 0 1; do
      local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null)
      [ "${u:-0}" -gt 2000 ] && busy=1
    done
    [ $busy -eq 0 ] && break
    sleep 10; w=$((w + 10))
  done
  sleep 5
}

for spec in "${ARMLIST[@]}"; do
  IFS='|' read -r name draft k window extra <<< "$spec"
  out="$PHASE/data/w10/w10_moe_${name}.json"
  if [ -s "$out" ] && grep -q '"complete": true' "$out"; then
    echo "[W10] skip $name (done)"; continue
  fi
  echo "[W10] === $name ==="
  cleanup
  env CUDA_VISIBLE_DEVICES=0,1 \
      G93_MODEL="$MODEL" G93_TP=2 G93_DRAFT="$draft" G93_K="$k" \
      G93_WINDOW="$window" G93_BATCHES=1,8,32,64 G93_ITERS=4 \
      G93_CEILING=16384 G93_MAXLEN=24576 \
      G93_TUNE=0 G93_FIXED_LEN="$FLEN" \
      G93_OUT="$out" G93_TAG="w10_moe_${name}" \
      $extra \
      timeout 10800 .venv/bin/python research/93_c1_grid/scripts/run_grid.py \
        >> "$LOG/w10_moe_${name}.log" 2>&1 \
        || echo "[W10] FAILED $name (continuing)"
done
cleanup
echo "[W10] done: $(ls $PHASE/data/w10/*.json 2>/dev/null | wc -l)/5"
