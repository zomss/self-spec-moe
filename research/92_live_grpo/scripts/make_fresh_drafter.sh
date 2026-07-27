#!/bin/bash
# Build the refresh-ceiling drafter: RTN W4-sym quantization of a
# stage-1 policy dump (CPU, isolated llmcompressor venv from phase 75).
# Usage: bash make_fresh_drafter.sh <global_step_number>
set -euo pipefail
STEP=${1:?usage: make_fresh_drafter.sh <step>}
CKPT_DIR=${CKPT_DIR:-/data/smcho/ckpts/92_grpo_no-sd}
SRC="$CKPT_DIR/global_step_${STEP}/actor/huggingface"
OUT="$HOME/ckpts/92-drafter-step${STEP}-W4A16-INT4-sym"
[ -f "$SRC/config.json" ] || { echo "no dump at $SRC"; exit 1; }
[ -d "$OUT" ] && { echo "exists: $OUT"; exit 0; }

LC_VENV="${LC_VENV:-$HOME/.cache/eff_lc_venv}"
MK=/data/smcho/self-spec-moe/research/75_efficient_rollout_repro/scripts/make_wxa16_ckpt.py
HF_HOME=/data/smcho/huggingface SRC_MODEL="$SRC" BITS=4 SYM=1 OUT_DIR="$OUT" \
  "$LC_VENV/bin/python" "$MK"
echo "built: $OUT"
