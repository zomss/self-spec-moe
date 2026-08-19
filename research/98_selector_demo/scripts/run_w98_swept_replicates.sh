#!/usr/bin/env bash
# Replicates of the swept grid the selector is scored on.
#
# Every point in sections 40-43 is a single boot, and this record has been
# corrected by a replicate three times: sections 18/20 (equal work), 29 (the
# stock baseline) and 34, where a co-tenant produced a 47.6% error that was
# invisible in one boot and would have become a finding. The headline --
# 99.61% of omniscient, +1.97% over the best static -- currently rests on
# unreplicated numbers.
#
# Four points already carry a replicate from section 34 (LI b8, LIO b8, SS b8
# via _r2, LO b8). This covers the four that do not, plus a fresh stock boot
# each, since a same-session denominator is the discipline section 15
# established.
#
# Serial on one GPU: timing measurement.
set -euo pipefail
cd "$(dirname "$0")/../../.."
R=research/98_selector_demo/scripts/run_w98_refined_lo.py
GPU="${GPU:-0}"
ARMS="${ARMS:-off,base,skip4,skip8,w1024skip4}"

for spec in "LI 16" "LIO 16" "SS 32" "LO 16"; do
  set -- $spec
  out="research/98_selector_demo/data/g98_noinstr_${1,,}_b${2}_r2"
  mkdir -p "$out"
  echo "[rep] ${1} b${2} stock $(date -Is)"
  W98_LO_CELL="$1" W98_LO_BATCH="$2" W98_LO_ARMS=stock \
    .venv/bin/python "$R" --output-dir "$out" --gpu "$GPU" >/dev/null \
    || echo "[rep] ${1} b${2} stock FAILED"
  echo "[rep] ${1} b${2} armed $(date -Is)"
  W98_LO_NOINSTRUMENT=1 W98_LO_QUANT=1 W98_LO_CELL="$1" W98_LO_BATCH="$2" \
    W98_LO_ARMS="$ARMS" .venv/bin/python "$R" --output-dir "$out" --gpu "$GPU" \
    >/dev/null || echo "[rep] ${1} b${2} armed FAILED"
done
echo "[rep] COMPLETE $(date -Is)"
