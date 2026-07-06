#!/bin/bash
# Phase 66 driver: 2-node measurement sequence. Near-pool b32 points get
# their OWN launch (P64 lesson: the JSON is written at batch-loop end, so a
# thrashing b32 sharing a launch kills the resident rows too).
# Predicted pools (from P65 measured, per-token 192->96 KiB):
#   EP-routed shared ~473k tok/rank -> b12 42%, b24 84% resident; b32 112%.
#   fp8-replica shared ~232k -> b12 86% resident.
set -u
S=/h/v-sukmincho/self-spec-moe/research/66_shared_kv/scripts
LOG=/h/v-sukmincho/self-spec-moe/research/66_shared_kv/logs

run() { echo "=== $* ($(date +%H:%M:%S)) ==="; bash "$S/run_arm.sh" "$@"; }

# Arm A: EP-routed bf16 shared-KV draft.
run a_ep 4 12,24 2 2400 ""
run a_ep 2 12,24 2 2400 ""
run a_ep 4 32   1 3000 "_b32"
run a_ep 2 32   1 3000 "_b32"

# Arm B: fp8 full-replica comm-free draft (largest resident batch ~b12).
run b_rep 4 12  2 2400 ""

# Arm C (stretch): node-local draft at the serving batch.
run c_nl 4 24   2 2400 ""
run c_nl 4 32   1 3000 "_b32"

echo "=== ALL DONE ($(date +%H:%M:%S)) ==="
