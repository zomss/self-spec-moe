#!/bin/bash
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"
source research/76_lever_latency_sweep/scripts/env_e76.sh
export CUDA_VISIBLE_DEVICES=0,1 NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo
unset VLLM_SELF_SPEC_COMPILE_CONSISTENT VLLM_SELF_SPEC_PROFILE
export VLLM_SELF_SPEC_DRAFT_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1
export VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1
export VLLM_SELF_SPEC_SHARED_KV=1 VLLM_SELF_SPEC_DRAFT_KV_WINDOW=512 VLLM_SELF_SPEC_DRAFT_KV_SINKS=16
export VLLM_SELF_SPEC_CPU_ORCH=1 VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE=1
export VLLM_SELF_SPEC_SKIP_PREFILL_DRAFT=1
export VLLM_SELF_SPEC_DRAFT_FULLCG=1 VLLM_SELF_SPEC_DRAFT_STEP0_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_WHOLECHAIN=1
export HF_HOME=/data/smcho/huggingface HF_HUB_OFFLINE=1 TMPDIR=/data/smcho/tmp
export COMPILE_MODEL=Qwen/Qwen3-32B COMPILE_DRAFT=/data/smcho/ckpts/Qwen3-32B-W4A16-INT4-gptq
export COMPILE_TP=2 COMPILE_BATCHES=1,8,16 COMPILE_CTXS=2000,8000,14000
export COMPILE_KV_LIMIT=200000 COMPILE_CELLS=policy_cells_32b.csv COMPILE_TABLE=policy_table_32b.json
for arm in ${C32_ARMS:-off k4 k5}; do
  echo "[32b-compile] arm=$arm ($(date +%H:%M:%S))"
  .venv/bin/python research/82_runtime_switching/scripts/compile_policy.py --measure $arm \
    > research/82_runtime_switching/logs/c32_$arm.log 2>&1 || echo "  $arm FAILED"
done
.venv/bin/python research/82_runtime_switching/scripts/compile_policy.py --solve
export E2_MODEL=Qwen/Qwen3-32B E2_DRAFT=/data/smcho/ckpts/Qwen3-32B-W4A16-INT4-gptq E2_TP=2
export E2_TRACE32=1 E2_POLICY="$REPO/research/82_runtime_switching/data/policy_table_32b.json"
export VLLM_SELF_SPEC_POLICY_FILE="$E2_POLICY"
export VLLM_SELF_SPEC_ACCEPT_PROBE_INTERVAL=128 VLLM_SELF_SPEC_ACCEPT_PROBE_BURST=8 VLLM_SELF_SPEC_ACCEPT_GATE_MIN_BATCH=1
for arm in off k4 k5 policy; do
  echo "[32b-trace] arm=$arm ($(date +%H:%M:%S))"
  if [ "$arm" = policy ]; then export VLLM_SELF_SPEC_POLICY_FILE="$E2_POLICY"; else unset VLLM_SELF_SPEC_POLICY_FILE; fi
  .venv/bin/python research/82_runtime_switching/scripts/e2_demo.py --arm $arm \
    > research/82_runtime_switching/logs/t32_$arm.log 2>&1 \
    && grep "\[E2\]" research/82_runtime_switching/logs/t32_$arm.log \
    || echo "  $arm FAILED (rc=$?)"
  cp research/82_runtime_switching/data/e2_${arm}.json \
     research/82_runtime_switching/data/t32_${arm}.json 2>/dev/null
done
echo "[32b] DONE ($(date +%H:%M:%S))"
