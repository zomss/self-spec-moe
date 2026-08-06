#!/usr/bin/env bash
# W8c: MLA under the SAME constant-batch protocol as W8b, so both
# arches are measured one way. W8a's MLA rows used natural EOS with
# 2048-token ceiling decodes (batch stays full, so they were sound) but
# mixed currencies for the cross-check: the profiled boot's step times
# vs W7's unperturbed serving rate. Here each profiled boot carries its
# OWN serving.json, making the absolute armed-step check internally
# consistent (P_ms vs b*tau*1e3/rate -- no T proxy involved).
# Two lanes: GPU0 = off x{1,8,32} (T), GPU1 = uncond-profiled x{1,8,32}.
set -uo pipefail
cd /data/smcho/self-spec-moe
PHASE=research/96_selector_foundations
LOG="$PHASE/logs"; mkdir -p "$LOG" "$PHASE/data/w8"

export PATH="/data/smcho/self-spec-moe/.venv/bin:$PATH"
export HF_HOME=/data/smcho/huggingface VLLM_CACHE_ROOT=/data/smcho/.cache/vllm
export TRITON_CACHE_DIR=/data/smcho/.cache/triton TMPDIR=/data/smcho/tmp

COMMON="VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1 VLLM_SELF_SPEC_CPU_ORCH=1"
SHARED="$COMMON VLLM_SELF_SPEC_SHARED_KV=1 VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE=1 VLLM_SELF_SPEC_SKIP_PREFILL_DRAFT=1 VLLM_SELF_SPEC_DRAFT_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1"
MODEL="deepseek-ai/DeepSeek-V2-Lite"; DRAFT="$HOME/ckpts/DeepSeek-V2-Lite-W8A16-INT8-chan"
FLEN=256

cleanup() {
  local g="$1"
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$g" 2>/dev/null); do
    kill -9 "$p" 2>/dev/null; done
  local w=0
  while [ $w -lt 180 ]; do
    local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null)
    [ "${u:-0}" -le 2000 ] && break; sleep 10; w=$((w + 10))
  done
  sleep 5
}

off_lane() {
  for b in 1 8 32; do
    out="$PHASE/data/w8/w8c_mla_off_b${b}.json"
    [ -f "$out" ] && continue
    cleanup 0
    env CUDA_VISIBLE_DEVICES=0 \
        G93_MODEL="$MODEL" G93_TP=1 G93_DRAFT=off G93_K=4 \
        G93_BATCHES="$b" G93_DATASETS=R2 G93_ITERS=3 \
        G93_CEILING=2048 G93_MAXLEN=24576 G93_TUNE=0 G93_FIXED_LEN=$FLEN \
        G93_OUT="$out" G93_TAG="w8c_mla_off_b$b" \
        timeout 1800 .venv/bin/python research/93_c1_grid/scripts/run_grid.py \
          >> "$LOG/w8c_mla_off_b$b.log" 2>&1 || echo "[W8c] FAILED off b$b"
  done
  cleanup 0
}

unc_lane() {
  for b in 1 8 32; do
    dir="$PHASE/data/w8/prof2_mla_b${b}_K4"
    [ -d "$dir" ] && [ -n "$(ls "$dir" 2>/dev/null)" ] && continue
    mkdir -p "$dir"
    cleanup 1
    env CUDA_VISIBLE_DEVICES=1 \
        VLLM_SELF_SPEC_PROFILE=1 VLLM_SELF_SPEC_PROFILE_OUT="$dir" \
        G93_MODEL="$MODEL" G93_TP=1 G93_DRAFT="$DRAFT" G93_K=4 \
        G93_BATCHES="$b" G93_DATASETS=R2 G93_ITERS=3 \
        G93_CEILING=2048 G93_MAXLEN=24576 G93_TUNE=0 G93_FIXED_LEN=$FLEN \
        G93_OUT="$dir/serving.json" G93_TAG="w8c_mla_uncond_b$b" \
        $SHARED \
        timeout 1800 .venv/bin/python research/93_c1_grid/scripts/run_grid.py \
          >> "$LOG/w8c_mla_uncond_b$b.log" 2>&1 || echo "[W8c] FAILED uncond b$b"
  done
  cleanup 1
}

off_lane & P0=$!
unc_lane & P1=$!
wait $P0 $P1
echo "[W8c] done: $(ls $PHASE/data/w8/w8c_mla_off_*.json 2>/dev/null | wc -l) off, $(ls -d $PHASE/data/w8/prof2_mla_* 2>/dev/null | wc -l) prof"
