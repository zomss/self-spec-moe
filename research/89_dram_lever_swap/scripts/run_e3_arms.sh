#!/bin/bash
# E3 arms: off (AR) | stale (drift, no refresh) | refresh (detector+swap)
cd /data/smcho/self-spec-moe
source research/76_lever_latency_sweep/scripts/env_e76.sh
unset VLLM_SELF_SPEC_COMPILE_CONSISTENT VLLM_SELF_SPEC_PROFILE
export VLLM_SELF_SPEC_DRAFT_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1
export VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1
export VLLM_SELF_SPEC_SHARED_KV=1 VLLM_SELF_SPEC_DRAFT_KV_WINDOW=512 VLLM_SELF_SPEC_DRAFT_KV_SINKS=16
export VLLM_SELF_SPEC_CPU_ORCH=1 VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE=1
export VLLM_SELF_SPEC_SKIP_PREFILL_DRAFT=1
export VLLM_SELF_SPEC_DRAFT_FULLCG=1 VLLM_SELF_SPEC_DRAFT_STEP0_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_WHOLECHAIN=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_DISABLED_KERNELS=MacheteLinearKernel,CutlassW4A8LinearKernel,AllSparkLinearKernel
export CUDA_VISIBLE_DEVICES=0
export E3_MODE=run E3_EPS=0,0.1,0.2,0.3,0.4 E3_GATE=3.6 E3_K=4

run_arm () {  # arm spec
  local arm=$1 spec=$2
  for try in 1 2; do
    echo "=== E3arm $arm try$try ($(date +%H:%M:%S)) ==="
    E3_ARM=$arm E3_SPEC=$spec \
      timeout -k 30 1500 .venv/bin/python research/89_dram_lever_swap/scripts/e3_rl_demo.py \
      && { echo "=== E3arm $arm OK ==="; return 0; }
    pkill -9 -f "scripts/e3_rl" 2>/dev/null; sleep 8
  done
  echo "=== E3arm $arm FALLBACK no-autotune ==="
  E3_ARM=$arm E3_SPEC=$spec R88_NO_AUTOTUNE=1 \
    timeout -k 30 1800 .venv/bin/python research/89_dram_lever_swap/scripts/e3_rl_demo.py \
    || echo "=== E3arm $arm GAVE UP ==="
}

run_arm off 0
run_arm stale 1
run_arm refresh 1
echo "=== E3 arms DONE ($(date +%H:%M:%S)) ==="

# appended (offset-safe): K2 arms -- the POLICY-SELECTED depth at this
# cell (b16 short-ctx T=1.0: compiled table + canonical R8 both pick
# K2, 1.05x vs AR; K4 loses 0.92x). Gate rescaled to K2 accept scale.
export E3_K=2 E3_GATE=2.45
run_arm stale_k2 stale 1
run_arm refresh_k2 refresh 1
echo "=== E3 K2 arms DONE ($(date +%H:%M:%S)) ==="
