#!/bin/bash
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"
source research/76_lever_latency_sweep/scripts/env_e76.sh
export CUDA_VISIBLE_DEVICES=7 NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo
unset VLLM_SELF_SPEC_COMPILE_CONSISTENT VLLM_SELF_SPEC_PROFILE
export VLLM_SELF_SPEC_DRAFT_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1
export VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1
export VLLM_SELF_SPEC_SHARED_KV=1 VLLM_SELF_SPEC_DRAFT_KV_WINDOW=512 VLLM_SELF_SPEC_DRAFT_KV_SINKS=16
export VLLM_SELF_SPEC_CPU_ORCH=1 VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE=1
export VLLM_SELF_SPEC_SKIP_PREFILL_DRAFT=1
export VLLM_SELF_SPEC_DRAFT_FULLCG=1 VLLM_SELF_SPEC_DRAFT_STEP0_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_WHOLECHAIN=1
export HF_HOME=/data/smcho/huggingface HF_HUB_OFFLINE=1 TMPDIR=/data/smcho/tmp
mv research/82_runtime_switching/data/policy_cells.csv research/82_runtime_switching/data/policy_cells_buggyskip.csv 2>/dev/null
mv research/82_runtime_switching/data/policy_table.json research/82_runtime_switching/data/policy_table_buggyskip.json 2>/dev/null
for arm in off k4 k6; do
  echo "[fx-compile] arm=$arm ($(date +%H:%M:%S))"
  .venv/bin/python research/82_runtime_switching/scripts/compile_policy.py --measure $arm \
    > research/82_runtime_switching/logs/cfx_$arm.log 2>&1 || echo "  $arm FAILED"
done
.venv/bin/python research/82_runtime_switching/scripts/compile_policy.py --solve
export E2_POLICY="$REPO/research/82_runtime_switching/data/policy_table.json"
for trace in default longmath longctx; do
  unset E2_LONGMATH E2_LONGCTX
  [ "$trace" = longmath ] && export E2_LONGMATH=1
  [ "$trace" = longctx ] && export E2_LONGCTX=1
  echo "=== TRACE $trace fixedskip ($(date +%H:%M:%S)) ==="
  bash research/82_runtime_switching/scripts/run_e2.sh off k4 k6 policy
  for arm in off k4 k6 policy; do
    cp "research/82_runtime_switching/data/e2_${arm}.json" \
       "research/82_runtime_switching/data/valfx_${trace}_${arm}.json" 2>/dev/null
  done
done
echo "=== FIXEDSKIP VALIDATION DONE ($(date +%H:%M:%S)) ==="
