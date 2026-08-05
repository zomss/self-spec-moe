#!/usr/bin/env bash
# E3 — gate arms (MLA, MoE). P7: the runtime must DISARM where the map says
# spec loses and ARM where it wins.
#
#   E95_GPU=0   bash research/95_c3_deploy/scripts/run_e3_gate.sh mla 0
#   E95_GPU=0,1 bash research/95_c3_deploy/scripts/run_e3_gate.sh moe 0
#
# Three arms per arch, all on the same batch sweep and datasets:
#   off     AR anchor
#   uncond  spec ALWAYS on at the map's best static config (no policy file)
#           -- this is what a deployment without a gate would ship, and the
#           measure of what the gate is worth
#   gated   same config + the compiled policy (K/OFF per step from C2)
#
# Why these arches are gate arms and not switching arms: their C2 tables show
# R > 1 below b32 (a draft step costs MORE than a target step -- MLA 1.17-2.29,
# MoE 1.08-1.70), so no depth can pay; R crosses below 1 only at b32, which is
# exactly where C1 Stage B found their only wins.
set -uo pipefail
cd /data/smcho/self-spec-moe

ARCH="${1:?usage: run_e3_gate.sh mla|moe [seed]}"
SEED="${2:-0}"
GPU="${E95_GPU:-0}"
PHASE=research/95_c3_deploy
LOG="$PHASE/logs"; mkdir -p "$LOG" "$PHASE/data"

export PATH="/data/smcho/self-spec-moe/.venv/bin:$PATH"
export HF_HOME=/data/smcho/huggingface
export VLLM_CACHE_ROOT=/data/smcho/.cache/vllm
export TRITON_CACHE_DIR=/data/smcho/.cache/triton
export TMPDIR=/data/smcho/tmp

COMMON="VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1 VLLM_SELF_SPEC_CPU_ORCH=1"
SHARED="$COMMON VLLM_SELF_SPEC_SHARED_KV=1 VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE=1 VLLM_SELF_SPEC_SKIP_PREFILL_DRAFT=1 VLLM_SELF_SPEC_DRAFT_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1"

case "$ARCH" in
  mla)
    MODEL="deepseek-ai/DeepSeek-V2-Lite"; TP=1; MAXLEN=24576
    # V2-Lite's base model LOOPS at T=0 on 8/9 datasets -> ceiling protocol (93)
    CEIL=2048
    DRAFT="$HOME/ckpts/DeepSeek-V2-Lite-W8A16-INT8-chan"
    TABLE="$PHASE/data/policy_mla_q-w8chan_wnone.json" ;;
  moe)
    MODEL="Qwen/Qwen3-30B-A3B"; TP=2; MAXLEN=24576; CEIL=16384
    DRAFT="$HOME/ckpts/Qwen3-30B-A3B-W4A16-INT4-sym"
    TABLE="$PHASE/data/policy_moe_q-w4a16_wnone.json" ;;
  *) echo "unknown arch $ARCH"; exit 2 ;;
esac
[ -f "$TABLE" ] || { echo "missing policy table $TABLE"; exit 2; }

BATCHES="${E95_BATCHES:-1,8,32,64}"
DATASETS="${E95_DATASETS:-R1,R2,R6}"
KMAX=4

# Kill leftovers, then GATE on memory actually being released. A fixed sleep
# is not enough: the mla/gated arm OOMed at KV-cache allocation because the
# previous engine's memory had not been returned 5s after kill -9 (the
# leftover-worker hazard recorded in the 93/94 ops notes). Wait for free
# memory instead of guessing.
cleanup() {
  for g in ${GPU//,/ }; do
    for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$g" 2>/dev/null); do
      kill -9 "$p" 2>/dev/null
    done
  done
  local waited=0
  while [ $waited -lt 180 ]; do
    local busy=0
    for g in ${GPU//,/ }; do
      local used
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null)
      [ "${used:-0}" -gt 2000 ] && busy=1
    done
    [ $busy -eq 0 ] && break
    sleep 10; waited=$((waited + 10))
  done
  echo "[E3] cleanup: GPUs free after ${waited}s"
  sleep 5
}

run_arm() {   # name  draft  extra-env
  local name="$1" draft="$2" extra="$3"
  local out="$PHASE/data/e3_${ARCH}_${name}_s${SEED}.json"
  if [ -f "$out" ]; then echo "[E3] skip $name (exists)"; return; fi
  echo "[E3] === $ARCH arm=$name seed=$SEED gpu=$GPU ==="
  cleanup
  env CUDA_VISIBLE_DEVICES="$GPU" \
      G93_MODEL="$MODEL" G93_TP="$TP" G93_DRAFT="$draft" G93_K="$KMAX" \
      G93_BATCHES="$BATCHES" G93_DATASETS="$DATASETS" G93_ITERS=3 \
      G93_CEILING="$CEIL" G93_MAXLEN="$MAXLEN" G93_OUT="$out" \
      G93_TAG="e3_${ARCH}_${name}" \
      $extra \
      timeout 3600 .venv/bin/python research/93_c1_grid/scripts/run_grid.py \
        >> "$LOG/e3_${ARCH}_s${SEED}.log" 2>&1
  [ $? -ne 0 ] && echo "[E3] FAILED $ARCH/$name (see $LOG/e3_${ARCH}_s${SEED}.log)"
}

run_arm off    off      ""
run_arm uncond "$DRAFT" "$SHARED"
run_arm gated  "$DRAFT" "$SHARED VLLM_SELF_SPEC_POLICY_FILE=$TABLE"
cleanup
echo "[E3] done arch=$ARCH seed=$SEED"
