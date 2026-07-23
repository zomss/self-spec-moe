#!/bin/bash
# E5d: live 2x2 on the drift trace -- {argmax, thompson} x {kmax3, kmax4}
# + AR anchor. All refresh-enabled, drain trace, GPU 6.
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
export CUDA_VISIBLE_DEVICES=6
export E3_MODE=run E3_EPS=0,0.1,0.2,0.3,0.4 E3_GATE=2.45 E3_POLL=10
export E3_MAXTOK=3072 E3_EOS=1
T3=research/89_dram_lever_swap/data/policy_table_hum_k3.json
T4=research/91_bandit_stage3/data/policy_table_hum_k4.json

run_arm () {  # name spec k table bandit
  local name=$1 spec=$2 k=$3 table=$4 bandit=$5
  for try in 1 2; do
    echo "=== E5d $name try$try ($(date +%H:%M:%S)) ==="
    env VLLM_SELF_SPEC_POLICY_FILE=$table VLLM_SELF_SPEC_BANDIT=$bandit \
      E3_ARM=$name E3_SPEC=$spec E3_K=$k E3_BG_DETECT=1 \
      timeout -k 30 1800 .venv/bin/python research/89_dram_lever_swap/scripts/e3_rl_demo.py && break
    pgrep -f "scripts/e3_rl_demo" | while read p; do kill -9 $p; done; sleep 8
  done
}

run_arm argmax_k3 1 3 $T3 0
run_arm bandit_k3 1 3 $T3 1
run_arm argmax_k4 1 4 $T4 0
run_arm bandit_k4 1 4 $T4 1
echo "=== E5d DONE ($(date +%H:%M:%S)) ==="
