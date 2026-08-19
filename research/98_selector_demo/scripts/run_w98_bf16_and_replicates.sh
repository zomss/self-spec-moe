#!/usr/bin/env bash
# Two jobs section 33 left open, both instrument-free.
#
# 1. The bf16 family at LI, LIO and SS. Every armed arm on the refined grid
#    is `w4a16-quantized`; that default is justified by measurement on LO
#    alone (section 25: 1.5678 against 1.3225 at equal work). Section 31
#    then showed a lever's SIGN appearing to flip between cells -- which
#    section 33 traced to the instrument, but the episode is the reason not
#    to assume a lever transfers. Quantization's leverage also shrinks with
#    context by arithmetic: it removes 54.5% of draft traffic at LO, 39.2%
#    at LIO, 35.4% at LI.
#
# 2. Replicates of every instrument-free point. Section 33's grid is single
#    boots, and this record has now had four published conclusions reversed
#    -- twice by a replicate, twice by finding a cost one side paid.
#
# bf16 runs first: it answers an open question, where the replicates confirm
# a measured one.
set -euo pipefail
cd "$(dirname "$0")/../../.."
R=research/98_selector_demo/scripts/run_w98_refined_lo.py
GPU="${GPU:-0}"

boot() {   # outdir cell batch envprefix arms
  local out="$1" cell="$2" batch="$3" mode="$4" arms="$5"
  mkdir -p "$out"
  if [ "$mode" = "quant" ]; then
    W98_LO_NOINSTRUMENT=1 W98_LO_QUANT=1 W98_LO_CELL="$cell" W98_LO_BATCH="$batch" \
      W98_LO_ARMS="$arms" .venv/bin/python "$R" --output-dir "$out" --gpu "$GPU" \
      >/dev/null || echo "  FAILED $out $arms"
  elif [ "$mode" = "sweep" ]; then
    W98_LO_NOINSTRUMENT=1 W98_LO_SWEEP=1 W98_LO_CELL="$cell" W98_LO_BATCH="$batch" \
      W98_LO_ARMS="$arms" .venv/bin/python "$R" --output-dir "$out" --gpu "$GPU" \
      >/dev/null || echo "  FAILED $out $arms"
  elif [ "$mode" = "bf16" ]; then
    W98_LO_NOINSTRUMENT=1 W98_LO_CELL="$cell" W98_LO_BATCH="$batch" \
      W98_LO_ARMS="$arms" .venv/bin/python "$R" --output-dir "$out" --gpu "$GPU" \
      >/dev/null || echo "  FAILED $out $arms"
  else
    W98_LO_CELL="$cell" W98_LO_BATCH="$batch" W98_LO_ARMS=stock \
      .venv/bin/python "$R" --output-dir "$out" --gpu "$GPU" \
      >/dev/null || echo "  FAILED $out stock"
  fi
}

echo "=== 1. bf16 family, instrument-free, b8 ==="
for cell in LI LIO SS; do
  out="research/98_selector_demo/data/g98_bf16_${cell,,}_b8"
  echo "[bf16] ${cell} $(date -Is)"
  boot "$out" "$cell" 8 stock ""
  boot "$out" "$cell" 8 bf16  off,base,skip4,w512skip4
  boot "$out" "$cell" 8 sweep w1024skip4
done

echo "=== 2. replicates of the instrument-free grid ==="
for spec in "LI 8" "LI 16" "LIO 8" "LIO 16" "SS 8"; do
  set -- $spec
  out="research/98_selector_demo/data/g98_noinstr_${1,,}_b${2}_r2"
  echo "[rep] ${1} b${2} $(date -Is)"
  boot "$out" "$1" "$2" stock ""
  boot "$out" "$1" "$2" quant off,base,skip4,w512skip4,w1024skip4
done
echo "[rep] LO 8 $(date -Is)"
boot research/98_selector_demo/data/g98_noinstr_lo_b8_r2 LO 8 stock ""
boot research/98_selector_demo/data/g98_noinstr_lo_b8_r2 LO 8 quant off,base,skip8,w1024skip4

echo "[all] COMPLETE $(date -Is)"
