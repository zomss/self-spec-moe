#!/usr/bin/env bash
# W7-FP4 (Stage 2): controlled precision comparison on DeepSeek-V2-Lite.
# Regime A = forced-PCIe + 100us emulated exposed per-collective A2A (comm-bound,
# the W7 headline). STRICTLY SERIAL: one DP=2 engine at a time (spec draft loop is
# CPU-bound; a concurrent engine inflates it ~7x). CUDA graphs ON, WARMUP=2,
# ITERS=3, OUTLEN=160/SHORTLEN=32 -- matches results_W7.md / results_W7_fp8.md
# exactly so the NVFP4 spec numbers sit next to the bf16/FP8 tables on identical
# methodology.
#
# Runs: nospec (bf16 target), bf16 full-replica spec, NVFP4 full-replica spec.
#   - nospec / bf16 use the bf16 checkpoint as both target and draft.
#   - nvfp4 keeps target bf16 but points the draft at the pre-quantized NVFP4 ckpt
#     (quantization auto-detected from its hf_quant_config.json -> modelopt_fp4).
set -uo pipefail
cd /data/smcho/ssm-w7fp4

PY=/data/smcho/self-spec-moe/.venv/bin/python
SCRIPT=research/34_worldA_system/scripts/w7_fp4_timing.py
BF16_MODEL=/home/smcho/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V2-Lite/snapshots/604d5664dddd88a0433dbae533b7fe9472482de0
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=1
export PYTHONPATH=/data/smcho/ssm-w7fp4
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0

export W7_BATCHES=${W7_BATCHES:-8,32,64,128,256}
export W7_KS=${W7_KS:-2,3,4}
export W7_OUTLEN=160
export W7_SHORTLEN=32
export W7_ITERS=3
export W7_WARMUP=2
export W7_GPU_MEM=0.85
export W7_EAGER=0
export W7_A2A_US=100
export W7_DP=2
export W7_TP=1
export W7_OUT=research/34_worldA_system/data
export W7_DRAFT_MODEL=${W7_DRAFT_MODEL:-/data/smcho/ssm-w7fp4/research/34_worldA_system/ckpts/dsv2lite-nvfp4}

wait_gpu_free() {
  local want=$((65*1024))
  for _ in $(seq 1 80); do
    local ok=1
    for d in ${CUDA_VISIBLE_DEVICES//,/ }; do
      local f
      f=$(nvidia-smi -i "$d" --query-gpu=memory.free \
          --format=csv,noheader,nounits 2>/dev/null)
      [ "${f:-0}" -lt "$want" ] && ok=0
    done
    [ "$ok" -eq 1 ] && return 0
    sleep 3
  done
  echo "WARN: visible GPUs not free after wait"
}

run() {  # mode tag draft_model
  local mode=$1 tag=$2 draft=$3
  export W7_TAG=$tag W7_DRAFT_MODEL=$draft
  wait_gpu_free
  echo "=== RUN mode=$mode tag=$tag draft=$draft $(date +%T) ==="
  $PY $SCRIPT "$mode" || echo "WARN: run mode=$mode tag=$tag exited nonzero"
}

# All three through the FP4 harness (same methodology); the draft model selects
# the precision: bf16 ckpt -> bf16 draft (W7 baseline), NVFP4 ckpt -> NVFP4 draft.
NVFP4_CKPT=${W7_DRAFT_MODEL:-/data/smcho/ssm-w7fp4/research/34_worldA_system/ckpts/dsv2lite-nvfp4}
run nospec v2lite_nospec "$BF16_MODEL"
run spec   v2lite_bf16   "$BF16_MODEL"
run spec   v2lite_nvfp4  "$NVFP4_CKPT"
echo "=== W7-FP4 V2-LITE SERIAL DONE $(date +%T) ==="
