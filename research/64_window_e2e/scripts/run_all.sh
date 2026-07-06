#!/bin/bash
# Phase 64 orchestrator: run the 4 mandatory arms SEQUENTIALLY (2 nodes = one
# EP16 engine at a time), in mission order. The optional eagle_k1 re-reference
# is launched separately if everything completes cleanly and time permits:
#   bash run_arm.sh eagle_k1 8,32 4 1500   # Phase 59 retry/timeout convention
set -u
S=/h/v-sukmincho/self-spec-moe/research/64_window_e2e/scripts

run() { echo "=================== $* ($(date +%H:%M:%S)) ==================="; "$@"; }

run bash "$S/run_arm.sh" nospec
run bash "$S/run_arm.sh" w512k2
run bash "$S/run_arm.sh" w512k4
run bash "$S/run_arm.sh" w256k4

echo "=================== ARMS 1-4 DONE ($(date +%H:%M:%S)) ==================="
