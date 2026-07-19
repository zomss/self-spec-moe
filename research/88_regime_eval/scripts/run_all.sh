#!/bin/bash
# Phase 88 E1/E2: all arms x all canonical regimes (8B, fixed stack).
set -e
cd /data/smcho/self-spec-moe
source research/76_lever_latency_sweep/scripts/env_e76.sh
unset VLLM_SELF_SPEC_COMPILE_CONSISTENT VLLM_SELF_SPEC_PROFILE
export VLLM_SELF_SPEC_DRAFT_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1
export VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1
export VLLM_SELF_SPEC_SHARED_KV=1 VLLM_SELF_SPEC_DRAFT_KV_WINDOW=512 VLLM_SELF_SPEC_DRAFT_KV_SINKS=16
export VLLM_SELF_SPEC_CPU_ORCH=1 VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE=1
export VLLM_SELF_SPEC_SKIP_PREFILL_DRAFT=1
export VLLM_SELF_SPEC_DRAFT_FULLCG=1 VLLM_SELF_SPEC_DRAFT_STEP0_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_WHOLECHAIN=1
export CUDA_VISIBLE_DEVICES=${R88_GPU:-0}

run_arm () {  # arm K extra-env...
  local arm=$1 k=$2; shift 2
  echo "=== R88 arm=$arm K=$k ($(date +%H:%M:%S)) ==="
  env "$@" R88_ARM=$arm R88_K=$k \
    R88_OUT=research/88_regime_eval/data/regimes_${arm}_k${k}.json \
    .venv/bin/python research/88_regime_eval/scripts/run_regimes.py
}

run_arm off 0
run_arm w4win 4
run_arm w4win 6
run_arm w4a8 6 VLLM_DISABLED_KERNELS=MacheteLinearKernel,CutlassW4A8LinearKernel,AllSparkLinearKernel
echo "=== R88 ALL DONE ($(date +%H:%M:%S)) ==="
