#!/usr/bin/env bash
# E5 -- the LO lattice re-measured under the registered request set.
#
# E4 showed n = batch (unbackfilled) carries +-23% occupancy noise and that
# the registered n = 4 x batch collapses it to 2.1. Section 45's lattice --
# every arm, every cell -- used n = batch, so its per-point winners and every
# claim derived from them need re-measurement.
#
# LO first: it is the rollout-shaped cell, where the noise was worst and
# where the +10% objective has to be evaluated. Twenty quantized arms plus
# stock and off, n = 32 against 8 concurrent, natural EOS.
set -euo pipefail
cd "$(dirname "$0")/../../.."
R=research/98_selector_demo/scripts/run_w98_refined_lo.py
GPU="${GPU:-0}"
out=research/101_rl_rollout_selector/data/e5_lattice_lo_b8_n32
mkdir -p "$out"
quiet=0
for i in $(seq 1 120); do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
  [ "$u" -lt 2000 ] && quiet=$((quiet+1)) || quiet=0
  [ "$quiet" -ge 2 ] && break
  sleep 20
done
echo "[e5] stock $(date -Is)"
W98_LO_N=32 W98_LO_CELL=LO W98_LO_BATCH=8 W98_LO_ARMS=stock \
  .venv/bin/python "$R" --output-dir "$out" --gpu "$GPU" >/dev/null 2>&1 \
  || echo "[e5] stock FAILED"
echo "[e5] lattice $(date -Is)"
# Quantized family only. Section 45 measured bf16 winning nothing at any of
# fourteen points and taking 2 of 70 top-5 slots, and its drafts are ~2x more
# expensive to boot, so the full 42-arm lattice would spend half its GPU time
# on arms already known to lose.
QUANT_ARMS="w_woff_skip0,w_woff_skip4,w_woff_skip8,w_woff_skip16,w_w128_skip0,w_w128_skip4,w_w128_skip8,w_w128_skip16,w_w256_skip0,w_w256_skip4,w_w256_skip8,w_w256_skip16,w_w512_skip0,w_w512_skip4,w_w512_skip8,w_w512_skip16,w_w1024_skip0,w_w1024_skip4,w_w1024_skip8,w_w1024_skip16"
W98_LO_N=32 W98_LO_NOINSTRUMENT=1 W98_LO_LATTICE=1 W98_LO_CELL=LO W98_LO_BATCH=8 \
  W98_LO_ARMS="off,${QUANT_ARMS}" \
  .venv/bin/python "$R" --output-dir "$out" --gpu "$GPU" >/dev/null 2>&1 \
  || echo "[e5] lattice PARTIAL -- resume by re-running"
n=$(ls "$out"/*.json 2>/dev/null | grep -vc summary || true)
echo "[e5] DONE ${n} records $(date -Is)"
