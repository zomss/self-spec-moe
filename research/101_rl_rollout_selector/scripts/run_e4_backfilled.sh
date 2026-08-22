#!/usr/bin/env bash
# E4 -- the registered request set: does backfill remove the draw variance?
#
# Phase 100 registers n = 4 x max batch, submitted at once, with continuous
# batching draining the set. Every measurement in phases 98-101 instead
# submitted exactly `batch` requests, so the batch drained 8 -> 1 unbackfilled
# and E3 measured a 49-point band on a single-draw arm comparison.
#
# Same two arms and the same four disjoint draws as E3, now with n = 32
# against 8 concurrent. Prediction: occupancy sd collapses from 23.3 and the
# throughput comparison converges on the equal-work answer (skip4 ahead by
# 5-9%, tracking tau). If the variance persists, queue depth is not the cause.
set -euo pipefail
cd "$(dirname "$0")/../../.."
R=research/98_selector_demo/scripts/run_w98_refined_lo.py
GPU="${GPU:-0}"
for off in 0 8 16 24; do
  out="research/101_rl_rollout_selector/data/e4_backfill_off${off}"
  mkdir -p "$out"
  for i in $(seq 1 90); do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
    [ "$u" -lt 2000 ] && break
    sleep 20
  done
  echo "[e4] prompts ${off}..$((off+31)), n=32 concurrent=8 $(date -Is)"
  W98_LO_N=32 W98_LO_PROMPT_OFFSET="$off" W98_LO_LATTICE=1 W98_LO_CELL=LO \
    W98_LO_BATCH=8 W98_LO_ARMS=w_w1024_skip4,w_w1024_skip8 \
    .venv/bin/python "$R" --output-dir "$out" --gpu "$GPU" >/dev/null 2>&1 \
    || echo "[e4] offset ${off} PARTIAL"
  echo "[e4] offset ${off} done $(date -Is)"
done
echo "[e4] COMPLETE $(date -Is)"
