#!/bin/bash
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"
TABLE=research/82_runtime_switching/data/policy_table.json
until [ -f "$TABLE" ]; do sleep 20; done
sleep 10
export E2_POLICY="$REPO/$TABLE" VLLM_SELF_SPEC_SKIP_PREFILL_DRAFT=1 E2_WHOLECHAIN=1
for trace in default longmath longctx; do
  unset E2_LONGMATH E2_LONGCTX
  [ "$trace" = longmath ] && export E2_LONGMATH=1
  [ "$trace" = longctx ] && export E2_LONGCTX=1
  echo "=== TRACE $trace policy-wc-recompiled ($(date +%H:%M:%S)) ==="
  bash research/82_runtime_switching/scripts/run_e2.sh policy
  cp research/82_runtime_switching/data/e2_policy.json \
     "research/82_runtime_switching/data/valwc2_${trace}_policy.json" 2>/dev/null
done
echo "=== RECOMPILED-POLICY VALIDATION DONE ==="
