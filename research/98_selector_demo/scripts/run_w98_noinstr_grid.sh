#!/usr/bin/env bash
# Section 32 follow-up: re-measure instrument-free the points section 31
# recorded with the profiler and koff trace enabled.
#
# Section 32 re-took LI, LIO and SS at batch 8 and every park verdict
# reversed. LI b16, LIO b16 and LO were not re-taken, so their section-31
# numbers remain instrumented -- which means published numbers are still
# scored against a baseline that does not pay the instrument.
#
# Stock is booted fresh in each directory rather than borrowed: it carries no
# instrument either way, but a same-session denominator is the discipline
# section 15 established when it checked the two families' parked boots
# against each other instead of assuming they matched.
set -euo pipefail
cd "$(dirname "$0")/../../.."
R=research/98_selector_demo/scripts/run_w98_refined_lo.py
GPU="${GPU:-0}"

run() {   # cell batch arms
  local cell="$1" batch="$2" arms="$3"
  local out="research/98_selector_demo/data/g98_noinstr_${cell,,}_b${batch}"
  mkdir -p "$out"
  echo "[ni2] ${cell} b${batch} stock $(date -Is)"
  W98_LO_CELL="$cell" W98_LO_BATCH="$batch" W98_LO_ARMS=stock \
    .venv/bin/python "$R" --output-dir "$out" --gpu "$GPU" >/dev/null \
    || echo "[ni2] ${cell} b${batch} stock FAILED"
  echo "[ni2] ${cell} b${batch} armed $(date -Is)"
  W98_LO_NOINSTRUMENT=1 W98_LO_CELL="$cell" W98_LO_QUANT=1 \
    W98_LO_BATCH="$batch" W98_LO_ARMS="$arms" \
    .venv/bin/python "$R" --output-dir "$out" --gpu "$GPU" >/dev/null \
    || echo "[ni2] ${cell} b${batch} armed FAILED"
}

run LI  16 off,base,skip4,w512skip4,w1024skip4
run LIO 16 off,base,skip4,w512skip4,w1024skip4
# LO's generations are 100x longer, so its arm set is trimmed to the ones the
# record actually reads: the parked reference, the unlevered arm, the deep
# skip section 25 corrected, and the top pick.
run LO   8 off,base,skip8,w1024skip4
echo "[ni2] COMPLETE $(date -Is)"
