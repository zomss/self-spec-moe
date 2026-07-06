#!/bin/bash
# Phase 64 orchestrator: SEQUENTIAL arms (2 nodes = one EP16 engine at a time).
#
# Restructured after the first w512k2 attempt: the self-spec draft_model path
# DUPLICATES per-token KV (measured 41.8 GiB pool / 228,256 tokens = 192
# KiB/token = 2x the model's 96 KiB), so at 16k the b32 point (32 x 16.3k =
# 522k tokens/rank) is 2.3x OVER the self-spec KV pool -> scheduler waves;
# it cannot finish inside the 1200s try budget that fits the resident points.
# So: (1) resident points per arm in one engine launch (b8; +b12 = 86% pool,
# the largest clean resident serving batch, for the headline arm + nospec);
# (2) the over-pool b32 point per arm as its OWN launch with TRY_TO=2700 and
# a _b32 tag suffix (reported with the Phase-59/63 over-pool caveat).
# The optional eagle_k1 re-reference is launched separately:
#   bash run_arm.sh eagle_k1 8,32 4 1500   # Phase 59 retry/timeout convention
set -u
S=/h/v-sukmincho/self-spec-moe/research/64_window_e2e/scripts

run() { echo "=================== $* ($(date +%H:%M:%S)) ==================="; "$@"; }

# -- denominator (resident at b32: nospec pool is 567k tokens/rank) --
run bash "$S/run_arm.sh" nospec 8,32 2 1200
run bash "$S/run_arm.sh" nospec 12 2 900 _b12

# -- self-spec resident points --
run bash "$S/run_arm.sh" w512k2 8 2 1200
run bash "$S/run_arm.sh" w512k4 8,12 2 1500
run bash "$S/run_arm.sh" w256k4 8 2 1200

# -- self-spec over-pool b32 points (waved; long budget, 1 try each) --
run bash "$S/run_arm.sh" w512k2 32 1 2700 _b32
run bash "$S/run_arm.sh" w512k4 32 1 2700 _b32
run bash "$S/run_arm.sh" w256k4 32 1 2700 _b32

echo "=================== ARMS DONE ($(date +%H:%M:%S)) ==================="
