#!/usr/bin/env bash
# E3 -- is the drain-occupancy advantage arm-intrinsic or prompt-set luck?
#
# w1024/skip8 wins the section-45 LO b8 rollout while losing every
# fixed-batch comparison. The mechanism is now identified: tokens per step is
# mean_active x tau, and skip8 trades 7.8% of tau for 13% more active
# requests, netting +4.9% tokens per step on top of a ~8% cheaper draft.
#
# What is untested is WHY it drains more slowly. Both arms are
# distribution-preserving but not bit-identical, so each stops each request
# somewhere slightly different. Either that is systematic (an exploitable
# lever effect, and occupancy belongs in the cost model) or it is the luck of
# this particular draw (and section 45's per-point winners carry an unbounded
# drain-luck component, which replicating BOOTS cannot detect because a
# replicate reuses the same prompts).
#
# Four disjoint slices of the frozen LO bundle, two arms, natural EOS, trace
# on so mean_active is readable from H_target_steps.
set -euo pipefail
cd "$(dirname "$0")/../../.."
R=research/98_selector_demo/scripts/run_w98_refined_lo.py
GPU="${GPU:-0}"
for off in 0 8 16 24; do
  out="research/101_rl_rollout_selector/data/e3_drain_off${off}"
  mkdir -p "$out"
  for i in $(seq 1 90); do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
    [ "$u" -lt 2000 ] && break
    sleep 20
  done
  echo "[e3] prompts ${off}-$((off+7)) $(date -Is)"
  W98_LO_PROMPT_OFFSET="$off" W98_LO_LATTICE=1 W98_LO_CELL=LO W98_LO_BATCH=8 \
    W98_LO_ARMS=w_w1024_skip4,w_w1024_skip8 \
    .venv/bin/python "$R" --output-dir "$out" --gpu "$GPU" >/dev/null 2>&1 \
    || echo "[e3] offset ${off} PARTIAL"
  echo "[e3] offset ${off} done $(date -Is)"
done
echo "[e3] COMPLETE $(date -Is)"
