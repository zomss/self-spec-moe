#!/usr/bin/env bash
# W7-FP8 (b): realistic model -- Qwen3-30B-A3B (128 experts, top-8, 48 layers).
# FP8 full-replica comm-free draft vs no-spec, forced-PCIe DP+EP, greedy, CUDA
# graphs ON. A few (batch,K) points. FP8 ~halves draft memory so the full replica
# may fit; if it OOMs the harness records the error per batch (engine stops) and
# we fall back (DP=4 or EP-shard draft) -- see results_W7_fp8.md.
#
# DP default 2 (matches the V2-Lite layout). Override W7_DP=4 to spread the target
# shard over 4 GPUs if DP=2 OOMs.
set -uo pipefail
cd /data/smcho/ssm-w7q

PY=/data/smcho/self-spec-moe/.venv/bin/python
SCRIPT=research/34_worldA_system/scripts/w7_fp8_timing.py
export NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=1
export PYTHONPATH=/data/smcho/ssm-w7q
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0

export W7_MODEL=/home/smcho/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39
export W7_DP=${W7_DP:-2}
export W7_TP=1
export W7_TRC=0
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export W7_BATCHES=${W7_BATCHES:-8,32,64}
export W7_KS=${W7_KS:-2,3}
export W7_OUTLEN=160
export W7_SHORTLEN=32
export W7_ITERS=3
export W7_WARMUP=2
export W7_GPU_MEM=${W7_GPU_MEM:-0.90}
export W7_EAGER=0
export W7_A2A_US=${W7_A2A_US:-100}
export W7_MAX_MODEL_LEN=2048
export W7_OUT=research/34_worldA_system/data

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

run() {  # mode tag draft_quant
  local mode=$1 tag=$2 dq=$3
  export W7_TAG=$tag W7_DRAFT_QUANT=$dq
  wait_gpu_free
  echo "=== RUN mode=$mode tag=$tag draft_quant=${dq:-bf16} dp=$W7_DP $(date +%T) ==="
  $PY $SCRIPT "$mode" || echo "WARN: run mode=$mode tag=$tag exited nonzero"
}

run nospec qwen30b_nospec ""
run spec   qwen30b_fp8    fp8
echo "=== W7-FP8 QWEN30B SERIAL DONE $(date +%T) ==="
