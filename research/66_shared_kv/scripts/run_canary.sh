#!/bin/bash
# Phase 66 validation rung 1: SHORT-ctx (~2k) single-node canary of the
# shared-KV drafter (bf16 EP-routed self-draft = the target itself).
# References (Phase 62 canary, DUPLICATED-KV drafter, same protocol):
#   canary_off (W=0, K=2):  accept_len 3.000 EXACTLY (bit-exact self-draft)
#   canary_w64 (W=64, K=2): accept_len 2.540
# Shared-KV must reproduce BOTH: any deviation = sharing bug (slot/overwrite
# discipline). Also log the KV pool: dup ref at this config = 217,696
# tokens/rank -> shared must be ~2x.
set -u
S=/h/v-sukmincho/self-spec-moe/research/66_shared_kv/scripts

common=(W66_K=2 W66_BATCHES=8 W66_CTX=2048 W66_MAXLEN=2560
        W66_QUANT= W66_REPLICA=0 W66_LOCALROUTE=0 W66_SHARED=1
        W66_ITERS=2 W66_WARMUP=1 W66_MNB=8192 W66_TRYTO=1200 W66_RETRIES=3)

echo "=== canary_off (shared-KV, window OFF) ==="
env "${common[@]}" W66_WINDOW=0  bash "$S/run_arm_1node.sh" canary_off 16600
echo "=== canary_w64 (shared-KV, window 64) ==="
env "${common[@]}" W66_WINDOW=64 bash "$S/run_arm_1node.sh" canary_w64 16700
echo "=== canary done ==="
