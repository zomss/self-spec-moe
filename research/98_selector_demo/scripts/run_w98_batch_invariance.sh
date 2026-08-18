#!/usr/bin/env bash
# Is acceptance batch-dependent, or was that content?
#
# Opening-of-run acceptance rises monotonically with batch -- skip8 0.6570 to
# 0.6934, w1024 0.8230 to 0.8464 over 2 to 16 -- at the same u, on the same
# model, under either stopping rule. But batch B means "the first B prompts"
# in this design, so the rise is equally consistent with later prompts simply
# being easier to draft.
#
# Replicating ONE prompt to fill the batch holds content exactly fixed. Any
# remaining batch dependence is then numerical: vLLM's kernels are not
# batch-invariant, and the draft runs at (B, 1) query positions while the
# target verifies at (B, K+1), so the two paths' reduction orders shift
# differently as batch grows and their argmaxes agree more or less often.
set -euo pipefail
cd "$(dirname "$0")/../../.."
ARMS="${ARMS:-skip8,w1024skip4}"
GEN="${GEN:-4096}"
for batch in ${BATCHES:-2 4 8 16}; do
  out="research/98_selector_demo/data/g98_lo_rep_b${batch}"
  mkdir -p "$out"
  echo "[rep] batch ${batch} gen ${GEN} -- ${ARMS} -- $(date -Is)"
  W98_LO_QUANT=1 W98_LO_BATCH="${batch}" W98_LO_ARMS="${ARMS}" \
    W98_LO_FIXED="${GEN}" W98_LO_REPLICATE=1 \
    .venv/bin/python research/98_selector_demo/scripts/run_w98_refined_lo.py \
    --output-dir "$out" --gpu "${GPU:-0}" || echo "[rep] batch ${batch} FAILED"
done
echo "[rep] COMPLETE $(date -Is)"
