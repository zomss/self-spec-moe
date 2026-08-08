#!/usr/bin/env bash
# W14/D0 — supervised single boot with a PROGRESS WATCHDOG.
#
# The draft graph-capture wedge manifests BEFORE the first measured cell,
# so a total timeout is the wrong instrument: it must cover a whole boot
# and therefore charges full price for every wedge. Measured healthy
# boot->first-cell on B: 41-55 s (AR), 128-160 s (speculative).
#
# The runner touches a heartbeat file at its first scored cell. If that
# file does not appear within W14D_WATCHDOG_S (default 400 s, ~2.5x the
# slowest healthy boot) the process is killed as a wedge. A watchdog kill
# is a CRASHED PROCESS, not an observation: it replaces its registered
# slot and never becomes data.
#
# usage: run_w14d_boot.sh <config> <K> <tag> <out.json> [extra env...]
set -uo pipefail
cd /data/smcho/self-spec-moe
PHASE=research/96_selector_foundations
LOG="$PHASE/logs/w14"; mkdir -p "$LOG" "$PHASE/data/w14"

export PATH="/data/smcho/self-spec-moe/.venv/bin:$PATH"
export HF_HOME=/data/smcho/huggingface VLLM_CACHE_ROOT=/data/smcho/.cache/vllm
export TRITON_CACHE_DIR=/data/smcho/.cache/triton TMPDIR=/data/smcho/tmp
export VLLM_DISABLED_KERNELS=MacheteLinearKernel,CutlassW4A8LinearKernel,AllSparkLinearKernel
export VLLM_USE_FLASHINFER_SAMPLER=0
export LD_LIBRARY_PATH="/usr/local/cuda-12/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
COMMON="VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1 VLLM_SELF_SPEC_CPU_ORCH=1"
SHARED="$COMMON VLLM_SELF_SPEC_SHARED_KV=1 VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE=1 VLLM_SELF_SPEC_SKIP_PREFILL_DRAFT=1 VLLM_SELF_SPEC_DRAFT_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1"

CONFIG="${1:?config}"; KK="${2:?K}"; TAG="${3:?tag}"; OUT="${4:?out}"
GPU="${W14_GPU:-1}"
WD="${W14D_WATCHDOG_S:-400}"
HB="/tmp/w14d_hb_$$"; rm -f "$HB"

cleanup() {
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$GPU" 2>/dev/null); do
    [ "$(ps -o user= -p "$p" 2>/dev/null)" = "smcho" ] || continue
    kill -9 "$p" 2>/dev/null
  done
  local w=0
  while [ $w -lt 180 ]; do
    local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null)
    [ "${u:-0}" -le 2000 ] && break; sleep 10; w=$((w+10))
  done
  sleep 5
}

if [ -s "$OUT" ] && grep -q '"complete": true' "$OUT"; then
  echo "[D] skip $TAG (done)"; exit 0
fi
cleanup
EXTRA=""; [ "$CONFIG" != "AR" ] && EXTRA="$SHARED"

env CUDA_VISIBLE_DEVICES="$GPU" \
    W14D_CONFIG="$CONFIG" W14D_K="$KK" W14D_TAG="$TAG" W14D_OUT="$OUT" \
    W14D_HEARTBEAT="$HB" \
    ${W14D_CTX:+W14D_CTX="$W14D_CTX"} ${W14D_RID:+W14D_RID="$W14D_RID"} \
    ${W14D_BATCH:+W14D_BATCH="$W14D_BATCH"} ${W14D_SMOKE:+W14D_SMOKE="$W14D_SMOKE"} \
    ${W14D_ITERS:+W14D_ITERS="$W14D_ITERS"} \
    $EXTRA \
    .venv/bin/python "$PHASE/scripts/w14d_measure.py" \
      >> "$LOG/${TAG}.log" 2>&1 &
PID=$!

# progress watchdog: heartbeat must appear within WD seconds
waited=0
while [ $waited -lt "$WD" ]; do
  kill -0 $PID 2>/dev/null || break          # exited on its own
  [ -f "$HB" ] && break                      # made progress
  sleep 5; waited=$((waited+5))
done
if [ ! -f "$HB" ] && kill -0 $PID 2>/dev/null; then
  echo "[D] WEDGE: $TAG made no progress in ${WD}s -- killing (not an observation)"
  kill -9 $PID 2>/dev/null; wait $PID 2>/dev/null
  rm -f "$HB"; cleanup; exit 75
fi
wait $PID; rc=$?
rm -f "$HB"
[ $rc -ne 0 ] && { echo "[D] FAILED $TAG rc=$rc"; exit $rc; }
echo "[D] ok $TAG"
