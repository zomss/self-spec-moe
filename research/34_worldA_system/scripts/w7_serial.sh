#!/usr/bin/env bash
# W7 STRICTLY SERIAL sweep. One DP=2 engine at a time on a single GPU pair with
# nothing else running -- the spec draft loop is CPU-orchestration-bound, so any
# concurrent engine inflates spec step time many-fold (measured). Run this alone.
#
# Regimes (CUDA graphs throughout; eager hides the comm signal under launch cost):
#   native : forced-PCIe, no emulated A2A (compute-bound on this small model)
#   a2a100 : forced-PCIe + 100us emulated exposed per-collective A2A (comm-bound)
#
# Usage: bash w7_serial.sh    (uses CUDA_VISIBLE_DEVICES, default 0,1)
set -euo pipefail
cd /data/smcho/self-spec-moe

PY=/data/smcho/self-spec-moe/.venv/bin/python
SCRIPT=research/34_worldA_system/scripts/w7_timing.py
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=1

export W7_BATCHES=${W7_BATCHES:-8,32,64,128,256}
export W7_KS=${W7_KS:-2,3,4,5}
export W7_OUTLEN=${W7_OUTLEN:-160}
export W7_SHORTLEN=${W7_SHORTLEN:-32}
export W7_ITERS=${W7_ITERS:-3}
# WARMUP>=2: ensures CUDA-graph capture for both output-length shapes completes
# before timing (warmup=1 occasionally leaked capture into a timed run for the
# heavier K, producing spuriously low decode times).
export W7_WARMUP=${W7_WARMUP:-2}
export W7_GPU_MEM=${W7_GPU_MEM:-0.85}
export W7_EAGER=0
export W7_OUT=research/34_worldA_system/data

# Orphaned vLLM workers can briefly hold GPU memory after a run exits. Wait until
# the visible GPUs are (nearly) free before starting the next engine, to avoid a
# startup memory race. Only inspects free memory; never kills processes.
wait_gpu_free() {
  local want=$((65*1024))  # MiB free wanted, on each visible GPU
  for _ in $(seq 1 60); do
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
  echo "WARN: visible GPUs ($CUDA_VISIBLE_DEVICES) not free after wait"
}

run() {  # mode regime a2a_us
  local mode=$1 regime=$2 a2a=$3
  export W7_A2A_US=$a2a
  wait_gpu_free
  echo "=== RUN mode=$mode regime=$regime a2a_us=$a2a $(date +%T) ==="
  $PY $SCRIPT "$mode" || echo "WARN: run mode=$mode regime=$regime exited nonzero"
}

# Emulated exposed-A2A regime first (the headline comm-bound case).
run nospec a2a100 100
run spec   a2a100 100
# Native regime (fat-fabric / compute-bound contrast).
run nospec native 0
run spec   native 0
echo "=== W7 SERIAL ALL DONE $(date +%T) ==="
