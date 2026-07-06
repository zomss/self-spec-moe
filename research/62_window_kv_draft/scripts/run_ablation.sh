#!/bin/bash
# Phase 62 Part 3: window-KV accept ablation at 16k ctx, Qwen3-30B-A3B,
# single-node DP8/EP8, batch 8, chat + on-dist prompts, K from $1 (default 4).
# Arms (sequential; one engine at a time):
#   A  canary-16k    bf16 EP-routed draft, window OFF        (expect ~K+1)
#   B1 window-only   bf16 EP-routed draft, WINDOW=1024
#   B2 window-only   bf16 EP-routed draft, WINDOW=512
#   B3 window-only   bf16 EP-routed draft, WINDOW=256
#   C  composed      fp8 FULL-REPLICA + local-route draft, WINDOW=512
#   D  quant-ref     fp8 FULL-REPLICA + local-route draft, window OFF
# Usage: run_ablation.sh [K]
set -u
K="${1:-4}"
S=/h/v-sukmincho/self-spec-moe/research/62_window_kv_draft/scripts

common=(W62_K="$K" W62_BATCHES=8 W62_CTX=16384 W62_MAXLEN=20480
        W62_ITERS=2 W62_WARMUP=1 W62_MNB=8192 W62_TRYTO=2400 W62_RETRIES=3)
bf16=(W62_QUANT= W62_REPLICA=0 W62_LOCALROUTE=0)
fp8=(W62_QUANT=fp8 W62_REPLICA=1 W62_LOCALROUTE=1)

run() { echo "=================== $1 ($(date +%H:%M:%S)) ==================="; \
        shift; "$@"; }

run A  env "${common[@]}" "${bf16[@]}" W62_WINDOW=0    bash "$S/run_arm.sh" A_full16k   14100
run B1 env "${common[@]}" "${bf16[@]}" W62_WINDOW=1024 bash "$S/run_arm.sh" B1_w1024    14200
run B2 env "${common[@]}" "${bf16[@]}" W62_WINDOW=512  bash "$S/run_arm.sh" B2_w512     14300
run B3 env "${common[@]}" "${bf16[@]}" W62_WINDOW=256  bash "$S/run_arm.sh" B3_w256     14400
run C  env "${common[@]}" "${fp8[@]}"  W62_WINDOW=512  bash "$S/run_arm.sh" C_fp8_w512  14500
run D  env "${common[@]}" "${fp8[@]}"  W62_WINDOW=0    bash "$S/run_arm.sh" D_fp8_full  14600
echo "=================== ABLATION DONE ($(date +%H:%M:%S)) ==================="
