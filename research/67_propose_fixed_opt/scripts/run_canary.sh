#!/bin/bash
# Phase 67 bit-exact gate: single-node 2k canary of the shared-KV self-draft
# (bf16 EP-routed = the target) with the step-0 decode compaction ON. Under
# shared KV the appended token reads verify's cached target-exact KV, so the
# q=1 compacted decode must reproduce the q=(K+2) step-0 EXACTLY:
#   W=0,  K=2 -> accept_len 3.000 EXACTLY (bit-exact self-draft; the gate)
#   W=64, K=2 -> accept_len 2.540 (Phase 62 windowed ref)
# Any deviation from the Phase-66 shared-KV refs (3.000 / 2.689) = compaction
# bug. Reuses Phase-66 run_arm_1node.sh with the new flag injected via EXTRA.
set -u
S66=/h/v-sukmincho/self-spec-moe/research/66_shared_kv/scripts
DEC="${W67_STEP0_DECODE:-1}"

common=(W66_K=2 W66_BATCHES=8 W66_CTX=2048 W66_MAXLEN=2560
        W66_QUANT= W66_REPLICA=0 W66_LOCALROUTE=0 W66_SHARED=1
        W66_ITERS=2 W66_WARMUP=1 W66_MNB=8192 W66_TRYTO=1200 W66_RETRIES=3
        W66_EXTRA="VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE=$DEC")

echo "=== canary_off (shared-KV, step0-decode=$DEC, window OFF) -> expect 3.000 ==="
env "${common[@]}" W66_WINDOW=0  bash "$S66/run_arm_1node.sh" p67can_off 16620
echo "=== canary_w64 (shared-KV, step0-decode=$DEC, window 64) -> expect ~2.69 ==="
env "${common[@]}" W66_WINDOW=64 bash "$S66/run_arm_1node.sh" p67can_w64 16720
echo "=== canary done ==="
