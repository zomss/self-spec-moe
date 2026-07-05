#!/bin/bash
# Phase 59 orchestrator: run the full context x mode matrix SEQUENTIALLY (2 nodes
# = one EP16 engine at a time). 2k is reused from Phase 57 (not re-run). For each
# of 16k and 32k: no-spec baseline, then EAGLE3 K1 & K2 (retry runner).
# Batches 8,32,64 fixed across contexts; iters=2 warmup=1; MNB=8192 (higher than
# 4096 -> ~2x prefill throughput at long ctx; workspace ~2 GiB at 30B, safe).
set -u
S=/h/v-sukmincho/self-spec-moe/research/59_multicontext/scripts
LOG=/h/v-sukmincho/self-spec-moe/research/59_multicontext/logs

run() { echo "=================== $* ($(date +%H:%M:%S)) ==================="; "$@"; }

# ---- 16k (MIDDLE) ----
run bash "$S/run_nospec_ctx.sh" 16384 17408 8,32,64 13700 2 1 8192
run bash "$S/run_eagle_ctx.sh"  16384 17408 8,32,64 "1 2" 13760 2 1 8192 1500 4

# ---- 32k (LONG) ----
run bash "$S/run_nospec_ctx.sh" 32768 33792 8,32,64 13710 2 1 8192
run bash "$S/run_eagle_ctx.sh"  32768 33792 8,32,64 "1 2" 13800 2 1 8192 2700 4

echo "=================== MATRIX DONE ($(date +%H:%M:%S)) ==================="
