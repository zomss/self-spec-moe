#!/usr/bin/env bash
# LO batch sweep, quantized family.
#
# Sections 14-17 pinned draft cost (1.1% on the arm that misses) and
# acceptance (5% against realized), and the composition is still 19% wrong on
# `woff/skip8`. What remains uncalibrated is the split between batch-SHARED
# and batch-PROPORTIONAL cost: section 7 solved it on the parked arm at one
# batch, where it is not identifiable. Batch is what separates them -- shared
# terms contribute 1/B per token, per-request terms contribute a constant --
# so this varies batch and holds everything else.
#
# Batch stops at 16 deliberately: at 32 the run needs ~544K KV tokens against
# this boot's ~355K, so requests would be preempted and recomputed and the
# measurement would carry that instead of the split.
set -euo pipefail
cd "$(dirname "$0")/../../.."
ARMS="${ARMS:-off,skip8,w1024skip4}"
for batch in ${BATCHES:-2 4 16}; do
  out="research/98_selector_demo/data/g98_lo_q4_b${batch}"
  mkdir -p "$out"
  echo "[sweep] batch ${batch} -- ${ARMS} -- $(date -Is)"
  W98_LO_QUANT=1 W98_LO_BATCH="${batch}" W98_LO_ARMS="${ARMS}" \
    .venv/bin/python research/98_selector_demo/scripts/run_w98_refined_lo.py \
    --output-dir "$out" --gpu "${GPU:-0}" || echo "[sweep] batch ${batch} FAILED"
done
echo "[sweep] COMPLETE $(date -Is)"
