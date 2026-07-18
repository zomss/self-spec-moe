#!/bin/bash
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"
export E2_POLICY="$REPO/research/82_runtime_switching/data/policy_table.json"
export VLLM_SELF_SPEC_SKIP_PREFILL_DRAFT=1
for trace in default longmath longctx; do
  unset E2_LONGMATH E2_LONGCTX
  [ "$trace" = longmath ] && export E2_LONGMATH=1
  [ "$trace" = longctx ] && export E2_LONGCTX=1
  echo "=== TRACE $trace policy-v2 ($(date +%H:%M:%S)) ==="
  bash research/82_runtime_switching/scripts/run_e2.sh policy
  cp research/82_runtime_switching/data/e2_policy.json \
     "research/82_runtime_switching/data/val_${trace}_policy2.json" 2>/dev/null
done
echo "=== POLICY-V2 DONE ==="
