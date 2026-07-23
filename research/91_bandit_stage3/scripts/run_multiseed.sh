#!/bin/bash
# GPU 1: multi-seed CIs -- (AR, argmax_k3, bandit_pf) x seeds {1,2}.
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
export CUDA_VISIBLE_DEVICES=1
export E3_MODE=run E3_EPS=0,0.1,0.2,0.3,0.4 E3_GATE=2.45 E3_POLL=10
export E3_MAXTOK=3072 E3_EOS=1
POL=research/89_dram_lever_swap/data/policy_table_hum_k3.json
for seed in 1 2; do
  echo "=== MS off seed$seed ($(date +%H:%M:%S)) ==="
  env E3_ARM=off_s$seed E3_SPEC=0 E3_K=0 E3_SEED=$seed \
    timeout -k 30 1800 .venv/bin/python research/89_dram_lever_swap/scripts/e3_rl_demo.py || true
  echo "=== MS argmax seed$seed ($(date +%H:%M:%S)) ==="
  env VLLM_SELF_SPEC_POLICY_FILE=$POL E3_ARM=argmax_s$seed E3_SPEC=1 E3_K=3 E3_SEED=$seed E3_BG_DETECT=1 \
    timeout -k 30 1800 .venv/bin/python research/89_dram_lever_swap/scripts/e3_rl_demo.py || true
  echo "=== MS bandit seed$seed ($(date +%H:%M:%S)) ==="
  env VLLM_SELF_SPEC_POLICY_FILE=$POL VLLM_SELF_SPEC_BANDIT=1 VLLM_SELF_SPEC_BANDIT_RESAMPLE=16 \
    E3_ARM=banditpf_s$seed E3_SPEC=1 E3_K=3 E3_SEED=$seed E3_BG_DETECT=1 \
    timeout -k 30 1800 .venv/bin/python research/89_dram_lever_swap/scripts/e3_rl_demo.py || true
done
echo "=== MS DONE ($(date +%H:%M:%S)) ==="
