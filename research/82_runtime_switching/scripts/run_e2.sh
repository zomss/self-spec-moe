#!/bin/bash
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"
source research/76_lever_latency_sweep/scripts/env_e76.sh
export CUDA_VISIBLE_DEVICES=${E2_GPU:-7} NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo
unset VLLM_SELF_SPEC_COMPILE_CONSISTENT VLLM_SELF_SPEC_PROFILE
export VLLM_SELF_SPEC_DRAFT_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1
export VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1
export VLLM_SELF_SPEC_SHARED_KV=1 VLLM_SELF_SPEC_DRAFT_KV_WINDOW=512 VLLM_SELF_SPEC_DRAFT_KV_SINKS=16
export VLLM_SELF_SPEC_CPU_ORCH=1 VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE=1
export HF_HOME=/data/smcho/huggingface HF_HUB_OFFLINE=1 TMPDIR=/data/smcho/tmp
for arm in "$@"; do
  echo "[E2-run] arm=$arm ($(date +%H:%M:%S))"
  if [ "$arm" = policy ]; then
    if [ -n "${E2_POLICY:-}" ]; then
      export VLLM_SELF_SPEC_POLICY_FILE="$E2_POLICY" E2_POLICY
      export VLLM_SELF_SPEC_ACCEPT_PROBE_INTERVAL=128 VLLM_SELF_SPEC_ACCEPT_PROBE_BURST=1 VLLM_SELF_SPEC_ACCEPT_GATE_MIN_BATCH=4
      unset VLLM_SELF_SPEC_ACCEPT_OFF_THRESHOLD VLLM_SELF_SPEC_SHORTCTX_OFF
    else
      export VLLM_SELF_SPEC_ACCEPT_OFF_THRESHOLD=0.84 VLLM_SELF_SPEC_ACCEPT_ON_THRESHOLD=0.86 VLLM_SELF_SPEC_ACCEPT_PROBE_INTERVAL=128 VLLM_SELF_SPEC_ACCEPT_PROBE_BURST=1 VLLM_SELF_SPEC_SHORTCTX_OFF=8000:8 VLLM_SELF_SPEC_ACCEPT_GATE_MIN_BATCH=4
    fi
  else
    unset VLLM_SELF_SPEC_ACCEPT_OFF_THRESHOLD VLLM_SELF_SPEC_POLICY_FILE
  fi
  .venv/bin/python research/82_runtime_switching/scripts/e2_demo.py --arm "$arm" \
    > "research/82_runtime_switching/logs/e2_${arm}.log" 2>&1 \
    && grep "\[E2\]" "research/82_runtime_switching/logs/e2_${arm}.log" \
    || echo "  [E2-run] $arm FAILED (rc=$?)"
done
echo "[E2-run] DONE"
