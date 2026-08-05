#!/usr/bin/env bash
# W6 item 2 driver: dense q-hum pool calibration.
# GPU0: (s-b2,w512), (s-b2,w2048), AR anchor. GPU1: (s-none,w512), (s-none,w2048).
set -uo pipefail
cd /data/smcho/self-spec-moe
PHASE=research/96_selector_foundations
LOG="$PHASE/logs"; mkdir -p "$LOG" "$PHASE/data/w6"

export PATH="/data/smcho/self-spec-moe/.venv/bin:$PATH"
export HF_HOME=/data/smcho/huggingface
export VLLM_CACHE_ROOT=/data/smcho/.cache/vllm
export TRITON_CACHE_DIR=/data/smcho/.cache/triton
export TMPDIR=/data/smcho/tmp
# piecewise realization (the certified deployment default), notune in-script
export VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1
export VLLM_SELF_SPEC_CPU_ORCH=1 VLLM_SELF_SPEC_SHARED_KV=1
export VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE=1 VLLM_SELF_SPEC_SKIP_PREFILL_DRAFT=1
export VLLM_SELF_SPEC_DRAFT_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1
# dense q-hum kernel selection (95/run_e0.sh)
export VLLM_DISABLED_KERNELS=MacheteLinearKernel,CutlassW4A8LinearKernel,AllSparkLinearKernel

cleanup() {
  local gpu="$1"
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$gpu" 2>/dev/null); do
    kill -9 "$p" 2>/dev/null
  done
  local waited=0
  while [ $waited -lt 180 ]; do
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" 2>/dev/null)
    [ "${used:-0}" -le 2000 ] && break
    sleep 10; waited=$((waited + 10))
  done
  sleep 5
}

run_boot() {  # gpu skip window
  local gpu="$1" skip="$2" win="$3"
  local tag="s-${skip//,/_}_w${win}"
  local out="$PHASE/data/w6/w6cal_dense_${tag}.json"
  if [ -f "$out" ]; then echo "[W6cal] skip $tag (exists)"; return; fi
  echo "[W6cal] === gpu=$gpu $tag ==="
  cleanup "$gpu"
  env CUDA_VISIBLE_DEVICES="$gpu" W6_SKIP="$skip" W6_WINDOW="$win" W6_OUT="$out" \
      timeout 3600 .venv/bin/python "$PHASE/scripts/run_w6_calib.py" \
        >> "$LOG/w6cal_${tag}.log" 2>&1
  [ $? -ne 0 ] && echo "[W6cal] FAILED $tag"
}

( run_boot 0 "2,8" 512
  run_boot 0 "2,8" 2048
  run_boot 0 "none" off ) &
L0=$!
( run_boot 1 "none" 512
  run_boot 1 "none" 2048 ) &
L1=$!
wait $L0 $L1
cleanup 0; cleanup 1
echo "[W6cal] calibration complete"
ls -la "$PHASE/data/w6/" | grep w6cal
