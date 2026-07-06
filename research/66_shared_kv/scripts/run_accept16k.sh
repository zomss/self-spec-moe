#!/bin/bash
# Phase 66 validation rung 3: single-node DP8/EP8 16k accept check (Phase 62
# protocol): W512 K=4 shared-KV bf16 EP-routed draft. Reference: Phase 62
# arm B2 (w512 K4, duplicated KV) accept_len 4.584; must land ~4.58.
# Rung 2 rides along: the engine log's "GPU KV cache size" must show a
# target-only pool (dup 16k ref at DP8 gpu0.90: 220,208 tokens/rank-ish;
# 2-node EP16 refs: dup 236,512 vs no-spec 567,136).
set -u
S=/h/v-sukmincho/self-spec-moe/research/66_shared_kv/scripts
env W66_K=4 W66_BATCHES=8 W66_CTX=16384 W66_MAXLEN=20480 W66_WINDOW=512 \
    W66_QUANT= W66_REPLICA=0 W66_LOCALROUTE=0 W66_SHARED=1 \
    W66_ITERS=2 W66_WARMUP=1 W66_MNB=8192 W66_TRYTO=1800 W66_RETRIES=3 \
    bash "$S/run_arm_1node.sh" accept16k 16800
