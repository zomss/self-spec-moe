#!/usr/bin/env bash
# W6 skip-identity stage, subset transfer test: does the per-layer
# acceptance-sensitivity RANKING transfer across regimes?
#
# 4 single-layer skips spanning depth (2, 8 = the known-good pair; 18
# mid; 30 late), dense q-hum, w512, uncond K4, regimes R5 + R1
# (contrasting content classes), seeds 0+1. Baseline f(none) already
# measured (w6cal_dense_s-none_w512.json).
#
# Pre-registered verdict rule:
#   sensitivity(L, regime) = f(none) - f(skip-L), per regime (seed-mean)
#   TRANSFERS iff the 4-layer rank order agrees across R5 and R1 with at
#   most one adjacent swap (Kendall tau >= 0.67) AND seed order agrees.
#   -> transfers: one-time per-arch L-boot profile licensed (values
#      reusable across regimes up to a per-regime scale)
#   -> fails: per-regime profiles required; G3 gated measurement mode
#      becomes the priority infra item.
set -uo pipefail
cd /data/smcho/self-spec-moe
PHASE=research/96_selector_foundations
LOG="$PHASE/logs"; mkdir -p "$LOG" "$PHASE/data/w6"

export PATH="/data/smcho/self-spec-moe/.venv/bin:$PATH"
export HF_HOME=/data/smcho/huggingface
export VLLM_CACHE_ROOT=/data/smcho/.cache/vllm
export TRITON_CACHE_DIR=/data/smcho/.cache/triton
export TMPDIR=/data/smcho/tmp
export VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1
export VLLM_SELF_SPEC_CPU_ORCH=1 VLLM_SELF_SPEC_SHARED_KV=1
export VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE=1 VLLM_SELF_SPEC_SKIP_PREFILL_DRAFT=1
export VLLM_SELF_SPEC_DRAFT_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1
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

run_layer() {  # gpu layer
  local gpu="$1" layer="$2"
  local out="$PHASE/data/w6/w6sens_dense_L${layer}.json"
  if [ -f "$out" ]; then echo "[W6sens] skip L$layer (exists)"; return; fi
  echo "[W6sens] === gpu=$gpu L$layer ==="
  cleanup "$gpu"
  env CUDA_VISIBLE_DEVICES="$gpu" W6_SKIP="$layer" W6_WINDOW=512 \
      W6_REGIMES="R5,R1" W6_OUT="$out" \
      timeout 2400 .venv/bin/python "$PHASE/scripts/run_w6_calib.py" \
        >> "$LOG/w6sens_L${layer}.log" 2>&1
  [ $? -ne 0 ] && echo "[W6sens] FAILED L$layer"
}

( run_layer 0 2
  run_layer 0 18 ) &
L0=$!
( run_layer 1 8
  run_layer 1 30 ) &
L1=$!
wait $L0 $L1
cleanup 0; cleanup 1
echo "[W6sens] transfer test complete"
