#!/bin/bash
# 32B gap-cell shallow probes (R4 summarization, R8 T=1.0 rollout):
# K2/K3 on both ckpts -- at 32B/TP2 the kernel winner may be w4win.
cd /data/smcho/self-spec-moe
source research/76_lever_latency_sweep/scripts/env_e76.sh
unset VLLM_SELF_SPEC_COMPILE_CONSISTENT VLLM_SELF_SPEC_PROFILE
export VLLM_SELF_SPEC_DRAFT_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1
export VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1
export VLLM_SELF_SPEC_SHARED_KV=1 VLLM_SELF_SPEC_DRAFT_KV_WINDOW=512 VLLM_SELF_SPEC_DRAFT_KV_SINKS=16
export VLLM_SELF_SPEC_CPU_ORCH=1 VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE=1
export VLLM_SELF_SPEC_SKIP_PREFILL_DRAFT=1
export VLLM_SELF_SPEC_DRAFT_FULLCG=1 VLLM_SELF_SPEC_DRAFT_STEP0_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_WHOLECHAIN=1
export CUDA_VISIBLE_DEVICES=${R88_GPUS:-0,1}
export R88_MODEL=Qwen/Qwen3-32B R88_TP=2
export R88_DRAFT_W4WIN=/data/smcho/ckpts/Qwen3-32B-W4A16-INT4-gptq
export R88_DRAFT_W4A8=/data/smcho/ckpts/Qwen3-32B-W4A8-gptq
export R88_REGIMES=R4,R8
HUM=VLLM_DISABLED_KERNELS=MacheteLinearKernel,CutlassW4A8LinearKernel,AllSparkLinearKernel

run_arm () {  # arm K extra-env...
  local arm=$1 k=$2; shift 2
  echo "=== R88-32B-shallow arm=$arm K=$k ($(date +%H:%M:%S)) ==="
  env "$@" R88_ARM=$arm R88_K=$k \
    R88_OUT=research/88_regime_eval/data/regimes32s_${arm}_k${k}.json \
    timeout -k 30 2400 .venv/bin/python research/88_regime_eval/scripts/run_regimes.py \
    || echo "=== R88-32B-shallow $arm K=$k FAILED rc=$? ==="
}

run_arm w4win 3
run_arm w4win 2
run_arm w4a8 3 $HUM
run_arm w4a8 2 $HUM
echo "=== R88-32B-shallow ALL DONE ($(date +%H:%M:%S)) ==="
