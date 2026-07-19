#!/bin/bash
# w4a8 arm retry: autotune-on with a 30-min cap, fallback autotune-off.
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
export VLLM_DISABLED_KERNELS=MacheteLinearKernel,CutlassW4A8LinearKernel,AllSparkLinearKernel
export R88_ARM=w4a8 R88_K=6
export R88_OUT=research/88_regime_eval/data/regimes_w4a8_k6.json

echo "=== R88 w4a8 retry, autotune ON, 30-min cap ($(date +%H:%M:%S)) ==="
timeout -k 30 1800 .venv/bin/python research/88_regime_eval/scripts/run_regimes.py
rc=$?
if [ $rc -eq 0 ]; then
  echo "=== R88 w4a8 DONE autotune-on ($(date +%H:%M:%S)) ==="
  exit 0
fi
echo "=== R88 w4a8 autotune-on failed rc=$rc; FALLBACK autotune OFF ($(date +%H:%M:%S)) ==="
sleep 20
R88_NO_AUTOTUNE=1 timeout -k 30 2400 .venv/bin/python research/88_regime_eval/scripts/run_regimes.py
rc=$?
echo "=== R88 w4a8 fallback rc=$rc ($(date +%H:%M:%S)) ==="
