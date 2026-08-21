#!/usr/bin/env bash
# Acceptance for the full quantized lattice, so the selector can be scored on
# the grid section 45 measured.
#
# Section 45 measured all 40 lattice arms as ground truth and found seven
# distinct winners across fourteen points, with 6.79% of switching value on
# the table. The selector cannot be scored against that grid because its
# prediction map covers five arms: thirty-five have no acceptance measurement.
#
# Round 1 does not reduce the work here. Its sound elimination rule kills only
# what cannot pay at PERFECT acceptance, which needs draft/verify > 3.93, and
# the most expensive arm in this lattice sits at 1.79-2.74. Nothing is
# eliminated at any cell, so Round 2 must measure every survivor -- all twenty
# quantized arms per cell. That is a result about the design, not a failure of
# this campaign, and it is why the campaign is this size.
#
# bf16 is excluded: section 45 measured it winning nothing at any of the
# fourteen points and taking 2 of 70 top-5 slots.
set -euo pipefail
cd "$(dirname "$0")/../../.."
P=research/98_selector_demo/scripts/probe_w98_long_u.py
GPU="${GPU:-0}"

# cell:generation budget, matched to each cell's natural output length
for spec in "SS:512" "LI:1024" "LIO:2048" "LO:8192"; do
  cell="${spec%%:*}"; gen="${spec#*:}"
  out="research/98_selector_demo/data/g98_latacc_${cell,,}_q4"
  mkdir -p "$out"
  for i in $(seq 1 60); do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
    [ "$u" -lt 2000 ] && break
    sleep 20
  done
  echo "[latacc] ${cell} gen=${gen} $(date -Is)"
  W98_LONGU_QUANT=1 W98_LONGU_LATTICE=1 W98_LONGU_CELL="$cell" \
    W98_LONGU_TOKENS="$gen" W98_LONGU_BATCH=8 \
    .venv/bin/python "$P" --output-dir "$out" --gpu "$GPU" >/dev/null 2>&1 \
    || echo "[latacc] ${cell} PARTIAL -- resume by re-running"
  n=$(ls "$out"/*.json 2>/dev/null | grep -vc summary || true)
  echo "[latacc] ${cell} DONE ${n}/20 arms $(date -Is)"
done
echo "[latacc] COMPLETE $(date -Is)"
