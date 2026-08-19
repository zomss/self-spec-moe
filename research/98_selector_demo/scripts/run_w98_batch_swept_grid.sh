#!/usr/bin/env bash
# Section 39's open question: is there a mix where selection is worth
# anything?
#
# The equal-mix score found the selector reaching 99.80% of omniscient while
# beating the best single static by 0.15% -- no cell wanted a meaningfully
# different lever. Section 33 pointed at the one place on this grid where the
# optimum demonstrably moves: LI's best arm changes IDENTITY between batches
# (woff/skip4 at b8, w1024/skip4 at b16) and LIO's margin widens with batch
# (1.232 -> 1.331). A batch-swept mix is where a selection case would have to
# come from.
#
# Completes the (cell, batch) grid to eight points on one common arm set:
# skip8 at the existing b16 cells, plus SS at b32 and LO at b16 which have no
# instrument-free measurement at all.
set -euo pipefail
cd "$(dirname "$0")/../../.."
R=research/98_selector_demo/scripts/run_w98_refined_lo.py
GPU="${GPU:-0}"
ARMS="${ARMS:-off,base,skip4,skip8,w1024skip4}"

fill() {  # cell batch arms outdir
  echo "[bs] $1 b$2 $3 $(date -Is)"
  W98_LO_NOINSTRUMENT=1 W98_LO_QUANT=1 W98_LO_CELL="$1" W98_LO_BATCH="$2" \
    W98_LO_ARMS="$3" .venv/bin/python "$R" --output-dir "$4" --gpu "$GPU" \
    >/dev/null || echo "[bs] $1 b$2 FAILED"
}
stock() {  # cell batch outdir
  echo "[bs] $1 b$2 stock $(date -Is)"
  W98_LO_CELL="$1" W98_LO_BATCH="$2" W98_LO_ARMS=stock \
    .venv/bin/python "$R" --output-dir "$3" --gpu "$GPU" >/dev/null \
    || echo "[bs] $1 b$2 stock FAILED"
}

# the two b16 cells already measured need only the fourth common arm
fill LI  16 skip8 research/98_selector_demo/data/g98_noinstr_li_b16
fill LIO 16 skip8 research/98_selector_demo/data/g98_noinstr_lio_b16
# SS at its registered high batch, and LO at b16 -- neither measured
for spec in "SS 32" "LO 16"; do
  set -- $spec
  out="research/98_selector_demo/data/g98_noinstr_${1,,}_b${2}"
  mkdir -p "$out"
  stock "$1" "$2" "$out"
  fill  "$1" "$2" "$ARMS" "$out"
done
echo "[bs] COMPLETE $(date -Is)"
