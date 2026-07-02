#!/usr/bin/env bash
# OV0a physics probe: forced-PCIe NCCL comm (and emulated _sleep) vs side-stream GEMM.
# Three configs: small batch (b64-shaped), serving batch (b512-shaped), and the
# emulated-latency variant (500us/coll) at b64 shape.
set -u
REPO=/data/smcho/self-spec-moe
cd "$REPO"
export PATH="$REPO/.venv/bin:$PATH"
export NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=1
D="$REPO/research/49_overlap_impl/data"
L="$REPO/research/49_overlap_impl/logs"
mkdir -p "$D" "$L"
PY="$REPO/.venv/bin/torchrun"
S="$REPO/research/49_overlap_impl/scripts/ov0a_overlap_bench.py"

# b64-shaped: global 64*(K+1)=192 verify tokens -> 24/rank; chain ~32 ms.
OV_TOKENS_PER_RANK=24 OV_TARGET_MS=32 OV_SLEEP_US=500 \
  OV_OUT="$D/ov0a_b64.json" \
  "$PY" --nproc_per_node=8 "$S" > "$L/ov0a_b64.log" 2>&1
echo "EXIT=$? b64"

# b512-shaped: global 512*3=1536 -> 192/rank; chain ~46 ms.
OV_TOKENS_PER_RANK=192 OV_TARGET_MS=46 OV_SLEEP_US=0 \
  OV_OUT="$D/ov0a_b512.json" \
  "$PY" --nproc_per_node=8 "$S" > "$L/ov0a_b512.log" 2>&1
echo "EXIT=$? b512"
echo "OV0A DONE"
