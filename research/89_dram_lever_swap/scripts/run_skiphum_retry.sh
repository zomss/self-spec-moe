#!/bin/bash
# skip x Humming arms: quiet-box retry (RKS,R1,R5cot on GPUs 4-5).
cd /data/smcho/self-spec-moe
source research/76_lever_latency_sweep/scripts/env_e76.sh
unset VLLM_SELF_SPEC_COMPILE_CONSISTENT VLLM_SELF_SPEC_PROFILE
export VLLM_SELF_SPEC_DRAFT_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1
export VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1
export VLLM_SELF_SPEC_SHARED_KV=1 VLLM_SELF_SPEC_DRAFT_KV_WINDOW=512 VLLM_SELF_SPEC_DRAFT_KV_SINKS=16
export VLLM_SELF_SPEC_CPU_ORCH=1 VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE=1
export VLLM_SELF_SPEC_SKIP_PREFILL_DRAFT=1
export VLLM_SELF_SPEC_DRAFT_FULLCG=1 VLLM_SELF_SPEC_DRAFT_STEP0_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_WHOLECHAIN=1
export CUDA_VISIBLE_DEVICES=4,5
export PYTHONFAULTHANDLER=1
export R88_MODEL=Qwen/Qwen3-32B R88_TP=2
export R88_DRAFT_W4A8=/data/smcho/ckpts/Qwen3-32B-W4A8-gptq
export VLLM_DISABLED_KERNELS=MacheteLinearKernel,CutlassW4A8LinearKernel,AllSparkLinearKernel
export VLLM_SELF_SPEC_DRAFT_SKIP_LAYERS=2,4,7,16
export R88_REGIMES=RKS,R1,R5cot R88_ARM=w4a8

clean_workers () {
  pkill -9 -f "run_regimes.py" 2>/dev/null; sleep 3
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
    cmd=$(ps -o cmd= -p ${p%,*} 2>/dev/null)
    case "$cmd" in *Worker_TP*) kill -9 ${p%,*} ;; esac
  done
  sleep 5
}

box_quiet () {  # co-tenant zone (GPUs 2-7) quiet; our own work on
  # GPUs 0-1 does not block the retry.
  local tot=0 i=0
  for m in $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); do
    case $i in 2|3|4|5|6|7) tot=$((tot + m));; esac
    i=$((i + 1))
  done
  [ $tot -lt 2000 ]
}

run_arm () {
  local k=$1
  for try in 1 2 3; do
    until box_quiet; do sleep 600; done
    echo "=== skiphum K=$k try$try ($(date +%H:%M:%S)) ==="
    R88_K=$k R88_OUT=research/89_dram_lever_swap/data/skip32_skiphum_k${k}.json \
      timeout -k 30 1500 .venv/bin/python research/88_regime_eval/scripts/run_regimes.py
    rc=$?
    [ $rc -eq 0 ] && { echo "=== skiphum K=$k OK ==="; return 0; }
    echo "=== skiphum K=$k try$try rc=$rc ==="
    clean_workers
  done
  echo "=== skiphum K=$k GAVE UP ==="
}

run_arm 5
run_arm 4
echo "=== skiphum retry DONE ($(date +%H:%M:%S)) ==="
