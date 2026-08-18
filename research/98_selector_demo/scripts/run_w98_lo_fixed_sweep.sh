#!/usr/bin/env bash
# Fixed-length batch sweep: batch varied, workload held identical.
#
# The natural-EOS sweep could not separate batch from context. At T=0 the arms
# still diverge -- batch-composition numerics flip near-tie argmaxes, so every
# armed arm differs from the parked one on 8 of 8 requests -- and under natural
# EOS that divergence moves where requests STOP. Generation length then varies
# by arm and by batch (skip8 ran 15,061 tokens at batch 8 against 18-19K
# elsewhere, shortest at exactly the batch where its speedup peaked), and
# length drives the drain, which is a first-order term. So "batch curvature"
# and "context growth" were confounded.
#
# ignore_eos removes the confound by construction: every arm at every batch
# emits the same number of tokens over the same prompts. 16384 is chosen to
# bracket the natural-EOS realized mean of 15,061-19,273, so this measures the
# same depth rather than a shallower one.
#
# Calibration, not a scored run -- the scored protocol keeps natural EOS.
set -euo pipefail
cd "$(dirname "$0")/../../.."
ARMS="${ARMS:-off,skip8,w1024skip4}"
GEN="${GEN:-16384}"
for batch in ${BATCHES:-2 4 8 16}; do
  out="research/98_selector_demo/data/g98_lo_fix_b${batch}"
  mkdir -p "$out"
  echo "[fix] batch ${batch} gen ${GEN} -- ${ARMS} -- $(date -Is)"
  W98_LO_QUANT=1 W98_LO_BATCH="${batch}" W98_LO_ARMS="${ARMS}" \
    W98_LO_FIXED="${GEN}" \
    .venv/bin/python research/98_selector_demo/scripts/run_w98_refined_lo.py \
    --output-dir "$out" --gpu "${GPU:-0}" || echo "[fix] batch ${batch} FAILED"
done
echo "[fix] COMPLETE $(date -Is)"
