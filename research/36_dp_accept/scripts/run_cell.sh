#!/usr/bin/env bash
# Run one isolation cell. Usage: run_cell.sh <tag> with AI_* env preset.
set -uo pipefail
PY=/data/smcho/self-spec-moe/.venv/bin/python
SCRIPT=/data/smcho/ssm-dp/research/36_dp_accept/scripts/accept_isolate.py
export PYTHONPATH=/data/smcho/ssm-dp
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
export HF_HOME=/home/smcho/.cache/huggingface
# Forced-PCIe trio for EP/DP runs
export NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=1
echo "=== CELL tag=$AI_TAG dp=$AI_DP quant=${AI_QUANT:-bf16} CVD=$CUDA_VISIBLE_DEVICES $(date +%T) ==="
$PY $SCRIPT
echo "=== CELL DONE tag=$AI_TAG $(date +%T) ==="
