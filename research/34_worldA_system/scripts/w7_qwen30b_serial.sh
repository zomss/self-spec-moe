#!/usr/bin/env bash
# W7-Qwen30B headline driver: fully-optimized World A self-spec stack vs no-spec on
# Qwen3-30B-A3B, attention-DP + EP (EP=DP), tp=1, REAL forced-PCIe (no A2A emulation).
#
# Full stack (spec): VLLM_SELF_SPEC_DRAFT_LOCAL_ROUTE=1 + DRAFT_FULL_REPLICA=1 +
# DRAFT_FULL_CG=1, draft FP8 (speculative_config.quantization=fp8), target bf16.
#
# Phases (env-selectable via W7Q_PHASE):
#   epscale : EP-scaling -- (batch=64, K=2) at DP=2, DP=4, DP=8.
#   sweep8  : full sweep at EP=8 (DP=8) -- batch {8,32,64,128,256} x K {2,3,4}.
#   all     : both (default).
# Each phase runs no-spec then spec, one engine (DP group) at a time.
set -uo pipefail
cd /data/smcho/self-spec-moe

PY=/data/smcho/self-spec-moe/.venv/bin/python
SCRIPT=research/34_worldA_system/scripts/w7_qwen30b_timing.py
export NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=1
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0

export W7_MODEL=/home/smcho/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39
export W7_TP=1
export W7_DRAFT_QUANT=fp8
export W7_FULL_CG=1
export W7_OUTLEN=160
export W7_SHORTLEN=32
export W7_ITERS=3
export W7_WARMUP=2
export W7_GPU_MEM=${W7_GPU_MEM:-0.90}
export W7_EAGER=0
export W7_MAX_MODEL_LEN=2048
export W7_OUT=research/34_worldA_system/data

PHASE=${W7Q_PHASE:-all}

devs_for_dp() {  # echo a CUDA_VISIBLE_DEVICES list of length $1
  local n=$1 i out=""
  for ((i=0;i<n;i++)); do out+="${i}"; [ $i -lt $((n-1)) ] && out+=","; done
  echo "$out"
}

wait_gpu_free() {  # wait until the visible GPUs each have >= 70GiB free
  local want=$((70*1024)) tries=${2:-100}
  for _ in $(seq 1 "$tries"); do
    local ok=1
    for d in ${1//,/ }; do
      local f
      f=$(nvidia-smi -i "$d" --query-gpu=memory.free \
          --format=csv,noheader,nounits 2>/dev/null)
      [ "${f:-0}" -lt "$want" ] && ok=0
    done
    [ "$ok" -eq 1 ] && return 0
    sleep 3
  done
  echo "WARN: visible GPUs ($1) not free after wait"
}

run() {  # dp mode tag batches ks full_cg
  local dp=$1 mode=$2 tag=$3 batches=$4 ks=$5 fcg=${6:-1}
  local devs; devs=$(devs_for_dp "$dp")
  export CUDA_VISIBLE_DEVICES=$devs
  export W7_DP=$dp W7_TAG=$tag W7_BATCHES=$batches W7_KS=$ks W7_FULL_CG=$fcg
  wait_gpu_free "$devs"
  echo "=== RUN dp=$dp ep=$dp mode=$mode tag=$tag batches=$batches ks=$ks full_cg=$fcg devs=$devs $(date +%T) ==="
  $PY $SCRIPT "$mode" || echo "WARN: dp=$dp mode=$mode tag=$tag exited nonzero"
}

# On Qwen3-30B (FlashAttention/FA3, non-MLA) the draft FULL-cudagraph captures
# stale attention -> accept_len collapses (~2.0 -> ~1.6 at K=2; isolated to
# FULL_CG, not FP8/skip-rebuild/sample-in-cg -- see results). So we measure spec
# BOTH ways: full stack as specified (FULL_CG=1, named headline) AND with FULL_CG
# off (PIECEWISE draft, correct accept) for the best-achievable number.
if [ "$PHASE" = "epscale" ] || [ "$PHASE" = "all" ]; then
  for dp in 2 4 8; do
    run "$dp" nospec epscale 64 2 1
    run "$dp" spec   epscale_fcg 64 2 1
    run "$dp" spec   epscale_pw  64 2 0
  done
fi

if [ "$PHASE" = "sweep8" ] || [ "$PHASE" = "all" ]; then
  run 8 nospec sweep8 8,32,64,128,256 2 1
  run 8 spec   sweep8_fcg 8,32,64,128,256 2,3,4 1
  run 8 spec   sweep8_pw  8,32,64,128,256 2,3,4 0
fi

echo "=== W7-QWEN30B SERIAL DONE phase=$PHASE $(date +%T) ==="
