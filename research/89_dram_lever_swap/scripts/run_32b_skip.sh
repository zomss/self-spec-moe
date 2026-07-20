#!/bin/bash
# Phase 89 side-quest (user 2026-07-20): MEASURE the 32B skip-set
# composition (skip{2,4,7,16} x W4-GPTQ x win512) -- upgrade the
# "predicted dominated" claim to measured. GPUs 4-5, TP2.
# Regimes: RKS (KnapSpec parity cell), R1 (b1 math), R5cot (b8 deep).
cd /data/smcho/self-spec-moe
source research/76_lever_latency_sweep/scripts/env_e76.sh
unset VLLM_SELF_SPEC_COMPILE_CONSISTENT VLLM_SELF_SPEC_PROFILE
export VLLM_SELF_SPEC_DRAFT_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1
export VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1
export VLLM_SELF_SPEC_SHARED_KV=1 VLLM_SELF_SPEC_DRAFT_KV_WINDOW=512 VLLM_SELF_SPEC_DRAFT_KV_SINKS=16
export VLLM_SELF_SPEC_CPU_ORCH=1 VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE=1
export VLLM_SELF_SPEC_SKIP_PREFILL_DRAFT=1
export VLLM_SELF_SPEC_DRAFT_FULLCG=1 VLLM_SELF_SPEC_DRAFT_STEP0_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_WHOLECHAIN=1
export CUDA_VISIBLE_DEVICES=${R89_GPUS:-4,5}
export PYTHONFAULTHANDLER=1
export R88_MODEL=Qwen/Qwen3-32B R88_TP=2
export R88_DRAFT_W4WIN=$HOME/ckpts/Qwen3-32B-W4A16-INT4-gptq
export R88_REGIMES=RKS,R1,R5cot

run_arm () {  # name arm K extra-env...
  local name=$1 arm=$2 k=$3; shift 3
  echo "=== R89-skip $name ($(date +%H:%M:%S)) ==="
  env "$@" R88_ARM=$arm R88_K=$k \
    R88_OUT=research/89_dram_lever_swap/data/skip32_${name}.json \
    timeout -k 30 2400 .venv/bin/python research/88_regime_eval/scripts/run_regimes.py \
    || echo "=== R89-skip $name FAILED rc=$? ==="
}

SKIP="VLLM_SELF_SPEC_DRAFT_SKIP_LAYERS=2,4,7,16"
run_arm off off 0
run_arm skipw4_k4 w4win 4 $SKIP
run_arm skipw4_k5 w4win 5 $SKIP
echo "=== R89-skip ALL DONE ($(date +%H:%M:%S)) ==="
