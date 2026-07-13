#!/bin/bash
# Second-distribution (math) beta sweep -- robustness check for paper sections
# 5/8. Claim-bearing arms at the map's central ctx (16384), all 3 archs in
# parallel on GPUs 0-2. Writes research/79_paper/data/beta.csv (math dist).
set -u
PHASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$PHASE/../.." && pwd)"
PY="$REPO/.venv/bin/python"
mkdir -p "$PHASE/logs"
export HF_HOME=/data/smcho/huggingface HF_HUB_OFFLINE=1 TMPDIR=/data/smcho/tmp

ARMS_DENSE="win512,skip50,kvq_fp8,q_int4,q_fp8"
ARMS_MOE="win512,skip50,kvq_fp8,lr25,q_int4,q_fp8"

run(){  # gpu model arms
  local GPU=$1 M=$2 ARMS=$3
  echo "[beta-math] $M (GPU $GPU, $(date +%H:%M:%S))"
  CUDA_VISIBLE_DEVICES=$GPU "$PY" "$PHASE/scripts/beta_dist2.py" \
    --model "$M" --ctx 16384 --arms "$ARMS" \
    > "$PHASE/logs/beta_math_${M}_c16384.log" 2>&1
  echo "[beta-math] $M done rc=$? ($(date +%H:%M:%S))"
}

run 0 dense "$ARMS_DENSE" &
run 1 moe   "$ARMS_MOE" &
run 2 mla   "$ARMS_MOE" &
wait
echo "[beta-math] ALL DONE ($(date +%H:%M:%S))"
