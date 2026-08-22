#!/usr/bin/env bash
# E2 -- acceptance re-measured under NATURAL EOS.
#
# Every acceptance number this project owns was measured under `ignore_eos`,
# and results_e1c_why.md section 4 measured what that costs: the ordering of
# arms reverses between filler and real content, and absolute acceptance is
# roughly halved. Every prediction map, shortlist score and schedule built on
# those curves inherits the defect.
#
# The honest cost of this protocol is thinner deep buckets: under natural EOS
# only the long tail of the length distribution reaches u > 3072, so bucket 3
# will carry far fewer armed steps than the ignore_eos campaign's. That is
# accepted -- section 35's truncation rule already says the deep buckets must
# not be read past the natural length distribution anyway.
set -euo pipefail
cd "$(dirname "$0")/../../.."
P=research/98_selector_demo/scripts/probe_w98_long_u.py
GPU="${GPU:-0}"
for spec in "LO:32768" "LI:2048" "LIO:4096" "SS:1024"; do
  cell="${spec%%:*}"; cap="${spec#*:}"
  out="research/101_rl_rollout_selector/data/e2_natural_${cell,,}_q4"
  mkdir -p "$out"
  for i in $(seq 1 90); do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
    [ "$u" -lt 2000 ] && break
    sleep 20
  done
  echo "[e2] ${cell} natural EOS, cap ${cap} $(date -Is)"
  W98_LONGU_IGNORE_EOS=0 W98_LONGU_QUANT=1 W98_LONGU_LATTICE=1 \
    W98_LONGU_CELL="$cell" W98_LONGU_TOKENS="$cap" W98_LONGU_BATCH=8 \
    .venv/bin/python "$P" --output-dir "$out" --gpu "$GPU" >/dev/null 2>&1 \
    || echo "[e2] ${cell} PARTIAL"
  n=$(ls "$out"/*.json 2>/dev/null | grep -vc summary || true)
  echo "[e2] ${cell} DONE ${n}/20 $(date -Is)"
done
echo "[e2] COMPLETE $(date -Is)"
