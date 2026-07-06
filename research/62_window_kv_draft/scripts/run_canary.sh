#!/bin/bash
# Phase 62 Part 2: correctness canary at SHORT (~2k) context, single node.
# Plain EP-routed bf16 self-draft (no local-route, no replica) = mathematically
# the target itself:
#   canary_off (W=0):  accept_len MUST be ~3.0 (K=2; greedy same-model draft).
#   canary_w64 (W=64): accept_len must drop measurably below 3.0 (window
#                      engages; window << 2k context) but stay > 2.0.
set -u
S=/h/v-sukmincho/self-spec-moe/research/62_window_kv_draft/scripts

common=(W62_K=2 W62_BATCHES=8 W62_CTX=2048 W62_MAXLEN=2560
        W62_QUANT= W62_REPLICA=0 W62_LOCALROUTE=0
        W62_ITERS=2 W62_WARMUP=1 W62_MNB=8192 W62_TRYTO=1200 W62_RETRIES=3)

echo "=== canary_off (window OFF) ==="
env "${common[@]}" W62_WINDOW=0  bash "$S/run_arm.sh" canary_off 13900
echo "=== canary_w64 (window 64) ==="
env "${common[@]}" W62_WINDOW=64 bash "$S/run_arm.sh" canary_w64 14000
echo "=== canary done ==="
