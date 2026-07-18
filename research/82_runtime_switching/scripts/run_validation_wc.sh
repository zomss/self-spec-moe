#!/bin/bash
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"
export E2_POLICY="$REPO/research/82_runtime_switching/data/policy_table.json"
export VLLM_SELF_SPEC_SKIP_PREFILL_DRAFT=1 E2_WHOLECHAIN=1
for trace in default longmath longctx; do
  unset E2_LONGMATH E2_LONGCTX
  [ "$trace" = longmath ] && export E2_LONGMATH=1
  [ "$trace" = longctx ] && export E2_LONGCTX=1
  echo "=== TRACE $trace wholechain ($(date +%H:%M:%S)) ==="
  bash research/82_runtime_switching/scripts/run_e2.sh k4 k6 policy
  for arm in k4 k6 policy; do
    cp "research/82_runtime_switching/data/e2_${arm}.json" \
       "research/82_runtime_switching/data/valwc_${trace}_${arm}.json" 2>/dev/null
  done
done
echo "=== WC VALIDATION DONE ($(date +%H:%M:%S)) ==="
