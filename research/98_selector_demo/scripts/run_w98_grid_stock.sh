#!/usr/bin/env bash
# Step 5 -- the refined grid against the deployment baseline.
#
# Section 30 measured the stock arm for the first time and overturned section
# 29: our runtime parked costs 17-22% on this box, so armed-versus-`off`
# ratios flatter every arm by that much, and against stock LI is lost while
# LIO is won only by the windowed arms.
#
# This extends that to the rest of the grid: b16 for LI and LIO, where
# Campaign 1's numbers are worse than at b8 (0.615-0.936), and the two cells
# that had no stock arm at all. LO's armed set is already measured at b8 under
# natural EOS, so only its denominator is missing.
#
# Stock is booted in its own invocation because the quantized-family flag maps
# every arm to the w4a16 weight version, which is meaningless for an arm that
# has no draft.
set -euo pipefail
cd "$(dirname "$0")/../../.."
RUNNER=research/98_selector_demo/scripts/run_w98_refined_lo.py
GPU="${GPU:-0}"
ARMED="${ARMED:-off,base,skip4,w512skip4,w1024skip4}"

run_cell() {   # cell batch outdir armlist
  local cell="$1" batch="$2" out="$3" arms="$4"
  mkdir -p "$out"
  echo "[grid] ${cell} b${batch} stock  $(date -Is)"
  W98_LO_CELL="$cell" W98_LO_BATCH="$batch" W98_LO_ARMS=stock \
    .venv/bin/python "$RUNNER" --output-dir "$out" --gpu "$GPU" >/dev/null \
    || echo "[grid] ${cell} b${batch} stock FAILED"
  if [ -n "$arms" ]; then
    echo "[grid] ${cell} b${batch} armed  $(date -Is)"
    W98_LO_CELL="$cell" W98_LO_QUANT=1 W98_LO_BATCH="$batch" W98_LO_ARMS="$arms" \
      .venv/bin/python "$RUNNER" --output-dir "$out" --gpu "$GPU" >/dev/null \
      || echo "[grid] ${cell} b${batch} armed FAILED"
  fi
}

run_cell LI  16 research/98_selector_demo/data/g98_arm_li_b16  "$ARMED"
run_cell LIO 16 research/98_selector_demo/data/g98_arm_lio_b16 "$ARMED"
run_cell SS   8 research/98_selector_demo/data/g98_arm_ss_b8   "$ARMED"
# LO's armed arms are already measured at b8 (g98_lo_q4); only stock is new.
run_cell LO   8 research/98_selector_demo/data/g98_arm_lo_b8   ""
echo "[grid] COMPLETE $(date -Is)"
