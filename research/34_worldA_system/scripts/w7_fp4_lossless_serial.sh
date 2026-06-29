#!/usr/bin/env bash
# W7-FP4 losslessness spot-check: nospec / bf16 / nvfp4, one engine at a time.
set -uo pipefail
cd /data/smcho/ssm-w7fp4

PY=/data/smcho/self-spec-moe/.venv/bin/python
SCRIPT=research/34_worldA_system/scripts/w7_fp4_lossless.py
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=1
export PYTHONPATH=/data/smcho/ssm-w7fp4
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0

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
}

for mode in nospec bf16 nvfp4; do
  wait_gpu_free
  echo "=== LOSSLESS mode=$mode $(date +%T) ==="
  $PY $SCRIPT "$mode" || echo "WARN: lossless mode=$mode nonzero"
done
echo "=== W7-FP4 LOSSLESS DONE $(date +%T) ==="
