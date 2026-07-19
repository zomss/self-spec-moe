#!/bin/bash
# Step-1 dig, R4 summarization: is the 512-tok draft window the accept
# killer? Arms: full-KV draft (window OFF) at K4/K2 on w4win and w4a8.
cd /data/smcho/self-spec-moe
source research/76_lever_latency_sweep/scripts/env_e76.sh
unset VLLM_SELF_SPEC_COMPILE_CONSISTENT VLLM_SELF_SPEC_PROFILE
export VLLM_SELF_SPEC_DRAFT_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1
export VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1
export VLLM_SELF_SPEC_SHARED_KV=1
export VLLM_SELF_SPEC_CPU_ORCH=1 VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE=1
export VLLM_SELF_SPEC_SKIP_PREFILL_DRAFT=1
export VLLM_SELF_SPEC_DRAFT_FULLCG=1 VLLM_SELF_SPEC_DRAFT_STEP0_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_WHOLECHAIN=1
export CUDA_VISIBLE_DEVICES=${R88_GPU:-1}
export R88_REGIMES=R4

run_arm () {  # name arm K window extra-env...
  local name=$1 arm=$2 k=$3 win=$4; shift 4
  echo "=== R4dig $name ($(date +%H:%M:%S)) ==="
  local wenv=()
  if [ "$win" != "none" ]; then
    wenv=(VLLM_SELF_SPEC_DRAFT_KV_WINDOW=$win VLLM_SELF_SPEC_DRAFT_KV_SINKS=16)
  fi
  env "${wenv[@]}" "$@" R88_ARM=$arm R88_K=$k \
    R88_OUT=research/88_regime_eval/data/r4dig_${name}.json \
    timeout -k 30 1800 .venv/bin/python research/88_regime_eval/scripts/run_regimes.py \
    || echo "=== R4dig $name FAILED rc=$? ==="
}

# NOTE: DRAFT_FULLCG requires a window (scratchpad materialises
# sinks+window keys) -- "full-KV draft" is realised as win8192, which
# covers the whole 8k article under the captured chain.
HUM=VLLM_DISABLED_KERNELS=MacheteLinearKernel,CutlassW4A8LinearKernel,AllSparkLinearKernel
run_arm w4_win8192_k4 w4win 4 8192
run_arm w4_win2048_k4 w4win 4 2048
run_arm w4_win512_k2 w4win 2 512
run_arm w4a8_win8192_k4 w4a8 4 8192 $HUM
echo "=== R4dig ALL DONE ($(date +%H:%M:%S)) ==="

# shallow-depth probe on the two gap regimes (added after K4-Humming
# measured R4 0.94x / R8 0.92x -- depth trend says K2/K3 may cross 1.0)
export R88_REGIMES=R4,R8
run_arm w4a8_win512_k3 w4a8 3 512 $HUM
run_arm w4a8_win512_k2 w4a8 2 512 $HUM
echo "=== R4dig shallow probes DONE ($(date +%H:%M:%S)) ==="
