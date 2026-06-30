#!/usr/bin/env bash
# Qwen3-30B DP=8 EP=8 forced-PCIe FULL-CG. MODE=spec|nospec
set -uo pipefail
PY=/data/smcho/self-spec-moe/.venv/bin/python
SCRIPT=/data/smcho/ssm-cc/research/37_compile_consistency/scripts/qwen30b_dp8.py
export PYTHONPATH=/data/smcho/ssm-cc
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
export HF_HOME=/home/smcho/.cache/huggingface
# forced-PCIe trio
export NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=1
export W7_DP=${W7_DP:-8} W7_K=${W7_K:-4} W7_BATCH=${W7_BATCH:-16} W7_OUTLEN=${W7_OUTLEN:-96}
export W7_GPU_MEM=${W7_GPU_MEM:-0.88}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export MODE=${MODE:-spec}
echo "=== Qwen3-30B DP=$W7_DP MODE=$MODE $(date +%T) ==="
$PY $SCRIPT
echo "=== DONE MODE=$MODE rc=$? $(date +%T) ==="
