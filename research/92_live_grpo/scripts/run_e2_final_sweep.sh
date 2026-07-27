#!/bin/bash
# Phase 92 E2: the standardized staleness curve (final numbers).
#
# 64 prompts x 2 seeds per point over the checkpoint ladder, K=4,
# T=1.0, training-prompt distribution. The earlier 16-prompt E1 rows
# were pipeline validation only.
#
# E2_FRESH=1 adds accept(drafter@k, policy@k) refresh-ceiling arms for
# steps that have a matching drafter ckpt (built via make_fresh_drafter.sh).
set -euo pipefail

PHASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO=/data/smcho/self-spec-moe
CKPT_DIR=${CKPT_DIR:-/data/smcho/ckpts/92_grpo_no-sd}
DRAFT0=${DRAFT0:-$HOME/ckpts/Qwen2.5-7B-W4A16-INT4-sym}
STEPS=${STEPS:-"1 2 4 8 12 16"}
SEEDS=${SEEDS:-"0 1"}
GPU=${E2_GPU:-0}

export HF_HOME=/data/smcho/huggingface
export CUDA_VISIBLE_DEVICES=$GPU
export PATH="$REPO/.venv/bin:$PATH"
cd "$REPO"

run_one() {  # <target> <draft|-> <tag> <seed>
  local target=$1 draft=$2 tag=$3 seed=$4
  local spec=1; [ "$draft" = "-" ] && spec=0
  echo "[e2] $tag seed=$seed"
  E92_TARGET="$target" E92_DRAFT="$draft" E92_SPEC=$spec E92_K=4 \
  E92_TAG="$tag" E92_SEED="$seed" E92_NPROMPTS=64 timeout 3600 \
    .venv/bin/python "$PHASE/scripts/e1_staleness_curve.py" \
    2>&1 | tee -a "$PHASE/logs/e2_sweep.log" | grep -E "\[E92\]" || true
  for pid in $(pgrep -u "$USER" -f "VLLM::EngineCore" || true); do
    kill "$pid" 2>/dev/null || true
  done
  sleep 5
}

for seed in $SEEDS; do
  run_one "Qwen/Qwen2.5-7B" "$DRAFT0" "e2-step0" "$seed"
  for s in $STEPS; do
    hf="$CKPT_DIR/global_step_${s}/actor/huggingface"
    [ -f "$hf/config.json" ] || { echo "[e2] SKIP step $s"; continue; }
    run_one "$hf" "$DRAFT0" "e2-step${s}-stale" "$seed"
    if [ "${E2_FRESH:-0}" = "1" ]; then
      dk="$HOME/ckpts/92-drafter-step${s}-W4A16-INT4-sym"
      [ -d "$dk" ] && run_one "$hf" "$dk" "e2-step${s}-fresh" "$seed"
    fi
  done
done
echo "[e2] sweep done -> $PHASE/data/e1_curve.jsonl"
