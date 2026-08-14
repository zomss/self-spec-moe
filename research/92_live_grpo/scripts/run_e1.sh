#!/bin/bash
# Phase 92 E1 driver: staleness curve over the stage-1 checkpoint ladder.
#
# For each global_step_k dump: accept(drafter@0, policy@k).
# Optionally (E1_FRESH=1, needs drafter@k ckpts built first): the
# refresh ceiling accept(drafter@k, policy@k).
#
# Usage: CKPT_DIR=/data/smcho/ckpts/92_grpo_no-sd bash run_e1.sh [steps...]
set -euo pipefail

PHASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO=/data/smcho/self-spec-moe
CKPT_DIR=${CKPT_DIR:-/data/smcho/ckpts/92_grpo_no-sd}
DRAFT0=${DRAFT0:-/data/smcho/ckpts/Qwen2.5-7B-W4A16-INT4-sym}
K=${E1_K:-4}
GPU=${E1_GPU:-0}
STEPS=("$@")

export HF_HOME=/data/smcho/huggingface
export CUDA_VISIBLE_DEVICES=$GPU
export PATH="$REPO/.venv/bin:$PATH"   # engine compile shells out to ninja
cd "$REPO"

run_one() {  # <target> <draft|-> <tag>
  local target=$1 draft=$2 tag=$3
  local spec=1; [ "$draft" = "-" ] && spec=0
  echo "[run_e1] $tag target=$target draft=$draft"
  E92_TARGET="$target" E92_DRAFT="$draft" E92_SPEC=$spec E92_K=$K \
  E92_TAG="$tag" timeout 3600 .venv/bin/python \
    "$PHASE/scripts/e1_staleness_curve.py" \
    2>&1 | tee -a "$PHASE/logs/e1_curve.log" | grep -E "\[E92\]|Error" || true
  # wedge guard: no orphaned engine may survive into the next boot
  for pid in $(pgrep -u "$USER" -f "VLLM::EngineCore" || true); do
    kill "$pid" 2>/dev/null || true
  done
  sleep 5
}

# anchor: step-0 == base model (drafter and target from the same weights)
run_one "Qwen/Qwen2.5-7B" "$DRAFT0" "step0"
run_one "Qwen/Qwen2.5-7B" "-" "step0-ar"

for s in "${STEPS[@]}"; do
  hf="$CKPT_DIR/global_step_${s}/actor/huggingface"
  if [ ! -f "$hf/config.json" ] || ! ls "$hf"/*.safetensors >/dev/null 2>&1; then
    echo "[run_e1] SKIP step $s: no hf dump at $hf"; continue
  fi
  run_one "$hf" "$DRAFT0" "step${s}-stale"
  if [ "${E1_FRESH:-0}" = "1" ]; then
    dk="/data/smcho/ckpts/92-drafter-step${s}-W4A16-INT4-sym"
    [ -d "$dk" ] && run_one "$hf" "$dk" "step${s}-fresh"
  fi
done
echo "[run_e1] done -> $PHASE/data/e1_curve.jsonl"
