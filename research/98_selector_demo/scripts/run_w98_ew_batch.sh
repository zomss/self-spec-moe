#!/usr/bin/env bash
# Equal-work cost against BATCH, for one arm.
#
# The cost model charges a step as (shared + B * per_request), which is linear
# in batch by construction. That is the assumption left standing after six
# refuted explanations for the batch-scaling error, and it has never been
# measured -- every cost arm in this phase ran at batch 8. The profiler
# reports `draft_chain_ms` and `verify_ms` per step directly, so sweeping
# batch measures the scaling instead of inferring it from throughput.
set -euo pipefail
cd "$(dirname "$0")/../../.."
ARM="${ARM:-w4a16-quantized/woff/skip8}"
for batch in ${BATCHES:-2 4 16}; do
  out="research/98_selector_demo/data/g98_ew_q4_b${batch}"
  mkdir -p "$out"
  echo "[ewb] batch ${batch} -- ${ARM} -- $(date -Is)"
  W98_EW_QUANT=1 W98_EW_BATCH="${batch}" W98_EW_ARMS="${ARM}" \
    .venv/bin/python research/98_selector_demo/scripts/run_w98_equalwork_cost.py \
    --output-dir "$out" --gpu "${GPU:-0}" || echo "[ewb] batch ${batch} FAILED"
done
echo "[ewb] COMPLETE $(date -Is)"
