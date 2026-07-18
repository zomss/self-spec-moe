#!/bin/bash
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"
TABLE=research/82_runtime_switching/data/policy_table.json
until [ -f "$TABLE" ]; do sleep 20; done
sleep 5
export E2_POLICY="$REPO/$TABLE" VLLM_SELF_SPEC_SKIP_PREFILL_DRAFT=1
for trace in default longmath longctx; do
  unset E2_LONGMATH E2_LONGCTX
  [ "$trace" = longmath ] && export E2_LONGMATH=1
  [ "$trace" = longctx ] && export E2_LONGCTX=1
  echo "=== TRACE $trace ($(date +%H:%M:%S)) ==="
  bash research/82_runtime_switching/scripts/run_e2.sh off k4 k6 policy
  for arm in off k4 k6 policy; do
    cp "research/82_runtime_switching/data/e2_${arm}.json" \
       "research/82_runtime_switching/data/val_${trace}_${arm}.json" 2>/dev/null
  done
done
echo "=== VALIDATION DONE ($(date +%H:%M:%S)) ==="
