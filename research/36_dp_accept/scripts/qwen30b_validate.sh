#!/usr/bin/env bash
# Qwen3-30B-A3B DP=2 self-spec accept: before vs after the W7-dp fix.
# FP8 full-replica comm-free draft, target bf16, forced-PCIe DP+EP, greedy.
# Usage:
#   FIX=0 -> baseline (draft compiled, the broken ~1.65)
#   FIX=1 -> VLLM_SELF_SPEC_DRAFT_EAGER=1 (uncompiled draft)
set -uo pipefail
PY=/data/smcho/self-spec-moe/.venv/bin/python
SCRIPT=/data/smcho/ssm-dp/research/36_dp_accept/scripts/accept_isolate.py
export PYTHONPATH=/data/smcho/ssm-dp
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
export HF_HOME=/home/smcho/.cache/huggingface
export NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=1

export AI_MODEL=/home/smcho/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39
export AI_DP=${AI_DP:-2} AI_TP=1 AI_TRC=1 AI_EP=1
export AI_K=${AI_K:-4} AI_BATCH=${AI_BATCH:-16} AI_OUTLEN=${AI_OUTLEN:-96}
export AI_QUANT=fp8 AI_FULLREP=1 AI_EAGER=0
export AI_GPU_MEM=${AI_GPU_MEM:-0.90} AI_MAXLEN=2048
export AI_OUT=/data/smcho/ssm-dp/research/36_dp_accept/data
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

if [ "${FIX:-0}" = "1" ]; then
  export VLLM_SELF_SPEC_DRAFT_EAGER=1
  export AI_TAG=qwen30b_dp${AI_DP}_fp8_FIX
else
  export VLLM_SELF_SPEC_DRAFT_EAGER=0
  export AI_TAG=qwen30b_dp${AI_DP}_fp8_baseline
fi

echo "=== Qwen3-30B DP=$AI_DP FIX=${FIX:-0} tag=$AI_TAG $(date +%T) ==="
$PY $SCRIPT
echo "=== DONE $AI_TAG $(date +%T) ==="
