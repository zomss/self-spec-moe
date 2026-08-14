#!/usr/bin/env bash
# W8b: constant-batch-width re-measurement for the MoE cost model.
# W8a's MoE rows are invalid: R2 decodes stop at ~200 tokens on natural
# EOS, so the batch DRAINS (32 -> 1) mid-run and both the e2e rate and
# the profiler's mean step time average over a collapsing width. MLA is
# unaffected (2048-token ceiling decodes keep the batch full) and is NOT
# rerun. Fix: G93_FIXED_LEN (ignore_eos + fixed max_tokens).
#
# 6 boots, all TP2 sequential: off x{1,8,32} (gives T) and
# uncond-profiled x{1,8,32} at K=4 (gives D, V, ovh, and the absolute
# armed-step cross-check P_ms vs b*tau*1e3/rate, which needs no T).
set -uo pipefail
cd /data/smcho/self-spec-moe
PHASE=research/96_selector_foundations
LOG="$PHASE/logs"; mkdir -p "$LOG" "$PHASE/data/w8"

export PATH="/data/smcho/self-spec-moe/.venv/bin:$PATH"
export HF_HOME=/data/smcho/huggingface VLLM_CACHE_ROOT=/data/smcho/.cache/vllm
export TRITON_CACHE_DIR=/data/smcho/.cache/triton TMPDIR=/data/smcho/tmp

COMMON="VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1 VLLM_SELF_SPEC_CPU_ORCH=1"
SHARED="$COMMON VLLM_SELF_SPEC_SHARED_KV=1 VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE=1 VLLM_SELF_SPEC_SKIP_PREFILL_DRAFT=1 VLLM_SELF_SPEC_DRAFT_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1"
MODEL="Qwen/Qwen3-30B-A3B"; DRAFT="/data/smcho/ckpts/Qwen3-30B-A3B-W4A16-INT4-sym"
FLEN=256

cleanup() {
  for g in 0 1; do
    for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$g" 2>/dev/null); do
      kill -9 "$p" 2>/dev/null; done
  done
  local w=0
  while [ $w -lt 180 ]; do
    local busy=0
    for g in 0 1; do
      local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null)
      [ "${u:-0}" -gt 2000 ] && busy=1
    done
    [ $busy -eq 0 ] && break
    sleep 10; w=$((w + 10))
  done
  sleep 5
}

for b in 1 8 32; do
  out="$PHASE/data/w8/w8b_moe_off_b${b}.json"
  if [ ! -f "$out" ]; then
    echo "[W8b] === off b=$b ==="
    cleanup
    env CUDA_VISIBLE_DEVICES=0,1 \
        G93_MODEL="$MODEL" G93_TP=2 G93_DRAFT=off G93_K=4 \
        G93_BATCHES="$b" G93_DATASETS=R2 G93_ITERS=3 \
        G93_CEILING=16384 G93_MAXLEN=24576 G93_TUNE=0 G93_FIXED_LEN=$FLEN \
        G93_OUT="$out" G93_TAG="w8b_moe_off_b$b" \
        timeout 3600 .venv/bin/python research/93_c1_grid/scripts/run_grid.py \
          >> "$LOG/w8b_moe_off_b$b.log" 2>&1 || echo "[W8b] FAILED off b$b"
  fi
done

for b in 1 8 32; do
  dir="$PHASE/data/w8/prof2_moe_b${b}_K4"
  if [ ! -d "$dir" ] || [ -z "$(ls "$dir" 2>/dev/null)" ]; then
    mkdir -p "$dir"
    echo "[W8b] === uncond-prof b=$b ==="
    cleanup
    env CUDA_VISIBLE_DEVICES=0,1 \
        VLLM_SELF_SPEC_PROFILE=1 VLLM_SELF_SPEC_PROFILE_OUT="$dir" \
        G93_MODEL="$MODEL" G93_TP=2 G93_DRAFT="$DRAFT" G93_K=4 \
        G93_BATCHES="$b" G93_DATASETS=R2 G93_ITERS=3 \
        G93_CEILING=16384 G93_MAXLEN=24576 G93_TUNE=0 G93_FIXED_LEN=$FLEN \
        G93_OUT="$dir/serving.json" G93_TAG="w8b_moe_uncond_b$b" \
        $SHARED \
        timeout 3600 .venv/bin/python research/93_c1_grid/scripts/run_grid.py \
          >> "$LOG/w8b_moe_uncond_b$b.log" 2>&1 || echo "[W8b] FAILED uncond b$b"
  fi
done
cleanup
echo "[W8b] done: $(ls $PHASE/data/w8/w8b_moe_off_*.json 2>/dev/null | wc -l) off, $(ls -d $PHASE/data/w8/prof2_* 2>/dev/null | wc -l) prof"
