#!/usr/bin/env bash
# W7-Qwen30B post-run confirmations (run AFTER the main sweep; uses all 8 GPUs):
#   1) comm-free: spec at DP=8 (EP=8) with VLLM_SELF_SPEC_LOG_A2A_COUNTS=1; grep the
#      per-rank "AgRs all2all count: ... real=N" at engine destroy. Full-replica draft
#      is non-EP (use_ep=False) -> structurally issues 0 collectives; the EP=8 VERIFY
#      issues the real all-to-all (real>0). One short spec engine, FULL_CG=1.
#   2) losslessness: greedy spec(fp8,FULL_CG) vs no-spec at DP=8, per-token agreement.
set -uo pipefail
cd /data/smcho/self-spec-moe
PY=/data/smcho/self-spec-moe/.venv/bin/python
export NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=1
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export W7_MODEL=/home/smcho/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39
export W7_DP=8 W7_TP=1 W7_MAX_MODEL_LEN=2048 W7_GPU_MEM=0.90 W7_OUT=research/34_worldA_system/data

echo "### COMM-FREE confirmation (DP=8, EP=8, spec FP8 FULL_CG, LOG_A2A=1) $(date +%T)"
W7_DRAFT_QUANT=fp8 W7_FULL_CG=1 W7_LOG_A2A=1 W7_TAG=commfree \
  W7_BATCHES=64 W7_KS=2 W7_OUTLEN=64 W7_SHORTLEN=32 W7_ITERS=1 W7_WARMUP=1 \
  $PY research/34_worldA_system/scripts/w7_qwen30b_timing.py spec 2>&1 \
  | grep -E "AgRs all2all count|use_ep=False|Draft model quantization" || true

echo "### LOSSLESS nospec $(date +%T)"
W7L_K=4 W7_DRAFT_QUANT=fp8 W7_FULL_CG=1 $PY research/34_worldA_system/scripts/w7_qwen30b_lossless.py nospec 2>&1 | grep -E "W7QL|Error|Traceback" || true
echo "### LOSSLESS fp8 (FULL_CG) $(date +%T)"
W7L_K=4 W7_DRAFT_QUANT=fp8 W7_FULL_CG=1 $PY research/34_worldA_system/scripts/w7_qwen30b_lossless.py fp8 2>&1 | grep -E "W7QL|Error|Traceback" || true

echo "### CONFIRM DONE $(date +%T)"
