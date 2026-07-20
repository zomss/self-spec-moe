#!/bin/bash
# E2a: recompile the 8B policy table on the WINNING lever (W4A8-Hum)
# with the shallow-K options included (off/k2/k3/k4/k6). GPU 0, TP1.
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
export CUDA_VISIBLE_DEVICES=0
export COMPILE_MODEL=Qwen/Qwen3-8B
export COMPILE_DRAFT=$HOME/ckpts/Qwen3-8B-W4A8-gptq
export VLLM_DISABLED_KERNELS=MacheteLinearKernel,CutlassW4A8LinearKernel,AllSparkLinearKernel
export COMPILE_CELLS=policy_cells_hum.csv
export COMPILE_TABLE=policy_table_hum.json
for arm in off k2 k3 k4 k6; do
  echo "[e2-compile] arm=$arm ($(date +%H:%M:%S))"
  .venv/bin/python research/82_runtime_switching/scripts/compile_policy.py --measure $arm
done
.venv/bin/python research/82_runtime_switching/scripts/compile_policy.py --solve
cp research/82_runtime_switching/data/policy_cells_hum.csv research/82_runtime_switching/data/policy_table_hum.json research/89_dram_lever_swap/data/ 2>/dev/null
echo "[e2-compile] DONE ($(date +%H:%M:%S))"
