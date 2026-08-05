#!/usr/bin/env bash
# E0 driver: one boot per (arch, window) + the AR anchor, on ONE gpu.
#
#   E95_GPU=0 bash research/95_c3_deploy/scripts/run_e0.sh dense 0
#   E95_GPU=1 bash research/95_c3_deploy/scripts/run_e0.sh llama 0
#
# args: <arch> <seed>.  Flags mirror the deployed C3 config (91/run_e6d.sh):
# FULLCG + wholechain + scratchpad + shared-KV, which is the stack every
# phase-94 R value was measured on.
set -u
cd /data/smcho/self-spec-moe

ARCH="${1:?arch}"
SEED="${2:-0}"
GPU="${E95_GPU:-0}"
PHASE=research/95_c3_deploy
LOG="$PHASE/logs"
mkdir -p "$LOG" "$PHASE/data"

export PATH="/data/smcho/self-spec-moe/.venv/bin:$PATH"
export HF_HOME=/data/smcho/huggingface
export VLLM_CACHE_ROOT=/data/smcho/.cache/vllm
export TRITON_CACHE_DIR=/data/smcho/.cache/triton
export TMPDIR=/data/smcho/tmp
export CUDA_VISIBLE_DEVICES="$GPU"

# deployed C3 stack
export VLLM_SELF_SPEC_DRAFT_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1
export VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1
export VLLM_SELF_SPEC_SHARED_KV=1 VLLM_SELF_SPEC_DRAFT_KV_SINKS=16
export VLLM_SELF_SPEC_CPU_ORCH=1 VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE=1
export VLLM_SELF_SPEC_SKIP_PREFILL_DRAFT=1
export VLLM_SELF_SPEC_DRAFT_FULLCG=1 VLLM_SELF_SPEC_DRAFT_STEP0_FULL_CG=1
export VLLM_SELF_SPEC_DRAFT_WHOLECHAIN=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

# dense's q-hum arm needs the Humming kernel selected (94/run_fullspace_audit.sh)
if [ "$ARCH" = "dense" ]; then
  export VLLM_DISABLED_KERNELS=MacheteLinearKernel,CutlassW4A8LinearKernel,AllSparkLinearKernel
fi

# leftover workers hold GPU memory after a parent dies and crash the next
# boot's free-mem check (93/94 ops lesson); kill by pid list, never pkill -f
# from inside this script (it matches its own cmdline).
cleanup() {
  pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$GPU" 2>/dev/null)
  for p in $pids; do kill -9 "$p" 2>/dev/null; done
  sleep 5
}

for W in off 512 2048 none; do
  OUT="$PHASE/data/e0_${ARCH}_w${W}_s${SEED}.json"
  if [ -f "$OUT" ]; then echo "[E0] skip $OUT (exists)"; continue; fi
  echo "[E0] === $ARCH window=$W seed=$SEED gpu=$GPU ==="
  cleanup
  E95_ARCH="$ARCH" E95_WINDOW="$W" E95_SEED="$SEED" E95_OUT="$OUT" \
    timeout 2400 .venv/bin/python "$PHASE/scripts/run_e0_envelope.py" \
      >> "$LOG/e0_${ARCH}_s${SEED}.log" 2>&1
  rc=$?
  [ $rc -ne 0 ] && echo "[E0] FAILED arch=$ARCH w=$W rc=$rc (see $LOG)"
done
cleanup
echo "[E0] done arch=$ARCH seed=$SEED"
