#!/usr/bin/env bash
# W1 boot matrix driver: llama R5, {off, w2048} x {tune, notune} x 3 boots.
#
#   bash research/96_selector_foundations/scripts/run_w1.sh
#
# Lane layout: GPU0 = w2048 (spec) lane, GPU1 = off (AR anchor) lane --
# parallel lanes match the phase-95 environment that produced the I4 swing.
# Within a lane, tune/notune boots are INTERLEAVED (tune b0, notune b0,
# tune b1, ...) so slow drift (thermal, co-tenant load) cannot masquerade
# as the autotune contrast. Every boot snapshots all-GPU clocks/util into
# its JSON (P-W1c arbitration).
set -uo pipefail
cd /data/smcho/self-spec-moe

PHASE=research/96_selector_foundations
LOG="$PHASE/logs"; mkdir -p "$LOG" "$PHASE/data/w1"

export PATH="/data/smcho/self-spec-moe/.venv/bin:$PATH"
export HF_HOME=/data/smcho/huggingface
export VLLM_CACHE_ROOT=/data/smcho/.cache/vllm
export TRITON_CACHE_DIR=/data/smcho/.cache/triton
export TMPDIR=/data/smcho/tmp

COMMON="VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1 VLLM_SELF_SPEC_CPU_ORCH=1"
SHARED="$COMMON VLLM_SELF_SPEC_SHARED_KV=1 VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE=1 VLLM_SELF_SPEC_SKIP_PREFILL_DRAFT=1 VLLM_SELF_SPEC_DRAFT_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1"

cleanup() {   # own GPU only; poll until memory actually released (95 hazard)
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

lane() {      # gpu  arm  (spec env only for the spec arm)
  local gpu="$1" arm="$2" env=""
  [ "$arm" != "off" ] && env="$SHARED"
  for boot in 0 1 2; do
    for tune in 1 0; do
      local tag="${arm}_$([ "$tune" = 1 ] && echo tune || echo notune)_b${boot}"
      local out="$PHASE/data/w1/w1_llama_${tag}.json"
      if [ -f "$out" ]; then echo "[W1] skip $tag (exists)"; continue; fi
      echo "[W1] === gpu=$gpu $tag ==="
      cleanup "$gpu"
      env CUDA_VISIBLE_DEVICES="$gpu" \
          W1_ARM="$arm" W1_TUNE="$tune" W1_BOOT="$boot" W1_OUT="$out" \
          $env \
          timeout 1800 .venv/bin/python \
            "$PHASE/scripts/run_w1_boot.py" \
            >> "$LOG/w1_${arm}.log" 2>&1
      [ $? -ne 0 ] && echo "[W1] FAILED $tag (see $LOG/w1_${arm}.log)"
    done
  done
  cleanup "$gpu"
  echo "[W1] lane done gpu=$gpu arm=$arm"
}

lane 0 w2048 &
L0=$!
lane 1 off &
L1=$!
wait $L0 $L1
echo "[W1] matrix complete"
ls -la "$PHASE/data/w1/"
