#!/bin/bash
# Phase 49 single-node native-NVLink no-spec reference (f-bound), DP8/EP8.
# Reuses the UNCHANGED Phase-47 harness; real fabric (no forced-PCIe env).
set -u
REPO=/h/v-sukmincho/self-spec-moe
PHASE=$REPO/research/52_two_node_e2e
PY=$REPO/.venv/bin/python
mkdir -p "$PHASE/logs" "$PHASE/data"

export PYTHONPATH="$REPO"
export HF_HUB_OFFLINE=1
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
export W7_MODEL=Qwen/Qwen3-30B-A3B
export W7_DP=8 W7_TP=1 W7_TRC=0 W7_EAGER=0
export W7_OUTLEN=160 W7_SHORTLEN=32 W7_ITERS=3 W7_WARMUP=2
export W7_GPU_MEM=0.90
export W7_BATCHES=32,64,128
export W7_OUT="$PHASE/data"
export W7_TAG=qwen30b_1node_nvlink

pkill -9 -f "w7_fp8_timing" 2>/dev/null; pkill -9 -f "EngineCore" 2>/dev/null
sleep 5
$PY "$REPO/research/34_worldA_system/scripts/w7_fp8_timing.py" nospec \
    > "$PHASE/logs/nospec_1node.log" 2>&1
RC=$?
pkill -9 -f "EngineCore" 2>/dev/null
echo "run_1node_ref done RC=$RC"
exit $RC
