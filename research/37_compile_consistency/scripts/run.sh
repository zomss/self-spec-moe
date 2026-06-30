#!/usr/bin/env bash
# Phase 37: compile-consistency. Generic runner for Qwen1.5-MoE DP1/DP2 accept.
set -uo pipefail
PY=/data/smcho/self-spec-moe/.venv/bin/python
SCRIPT=/data/smcho/ssm-cc/research/37_compile_consistency/scripts/accept_isolate.py
export PYTHONPATH=/data/smcho/ssm-cc
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
export HF_HOME=/home/smcho/.cache/huggingface
export NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=1

export AI_MODEL=${AI_MODEL:-/home/smcho/.cache/huggingface/hub/models--Qwen--Qwen1.5-MoE-A2.7B/snapshots/1a758c50ecb6350748b9ce0a99d2352fd9fc11c9}
export AI_DP=${AI_DP:-1} AI_TP=${AI_TP:-1} AI_TRC=${AI_TRC:-0} AI_EP=${AI_EP:-1}
export AI_K=${AI_K:-4} AI_BATCH=${AI_BATCH:-16} AI_OUTLEN=${AI_OUTLEN:-128}
export AI_QUANT=${AI_QUANT:-} AI_FULLREP=${AI_FULLREP:-1} AI_EAGER=${AI_EAGER:-0}
export AI_GPU_MEM=${AI_GPU_MEM:-0.85} AI_MAXLEN=${AI_MAXLEN:-2048}
export AI_OUT=/data/smcho/ssm-cc/research/37_compile_consistency/data
export AI_TAG=${AI_TAG:-run}

echo "=== START tag=$AI_TAG DP=$AI_DP K=$AI_K eager=$AI_EAGER $(date +%T) ==="
$PY $SCRIPT
echo "=== DONE $AI_TAG rc=$? $(date +%T) ==="
