#!/usr/bin/env bash
# W7 full sweep driver. Runs nospec + spec(K sweep) under several regimes on a
# DP=2 + EP forced-PCIe layout (DeepSeek-V2-Lite). All runs use CUDA graphs
# (W7_EAGER=0) -- the realistic serving config; eager hides the comm-bound
# signal under per-step launch overhead.
#
# Regimes:
#   native (W7_A2A_US=0)  : bare forced-PCIe fabric (compute-bound on this small model)
#   a2a100  (100 us)      : emulated exposed inter-node all-to-all (comm-bound target)
#
# Usage: bash w7_run_all.sh <regime>     regime in {native, a2a100, a2a50, eager_native}
set -euo pipefail
cd /data/smcho/self-spec-moe

PY=/data/smcho/self-spec-moe/.venv/bin/python
SCRIPT=research/34_worldA_system/scripts/w7_timing.py
DATA=research/34_worldA_system/data
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=1

export W7_BATCHES=${W7_BATCHES:-8,32,64,128,256}
export W7_KS=${W7_KS:-2,3,4,5}
export W7_OUTLEN=${W7_OUTLEN:-160}
export W7_SHORTLEN=${W7_SHORTLEN:-32}
export W7_ITERS=${W7_ITERS:-3}
export W7_WARMUP=${W7_WARMUP:-1}
export W7_GPU_MEM=${W7_GPU_MEM:-0.90}
export W7_OUT=$DATA

regime=${1:-native}
case "$regime" in
  native)       export W7_EAGER=0 W7_A2A_US=0 ;;
  a2a100)       export W7_EAGER=0 W7_A2A_US=100 ;;
  a2a50)        export W7_EAGER=0 W7_A2A_US=50 ;;
  eager_native) export W7_EAGER=1 W7_A2A_US=0 ;;
  *) echo "unknown regime $regime"; exit 1 ;;
esac

echo "=== W7 regime=$regime EAGER=$W7_EAGER A2A_US=$W7_A2A_US batches=$W7_BATCHES K=$W7_KS ==="
echo "=== NOSPEC ==="
$PY $SCRIPT nospec
echo "=== SPEC ==="
$PY $SCRIPT spec
echo "=== W7 regime=$regime DONE ==="
