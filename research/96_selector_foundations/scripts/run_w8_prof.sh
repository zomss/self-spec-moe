#!/usr/bin/env bash
# W8a: profiled uncond boots for the constructive cost model
# (w8_cost_model.md §3). arch x b{1,8,32} x K{1,4}, R2, notune,
# VLLM_SELF_SPEC_PROFILE coarse. MLA pair-parallel lanes; MoE TP2
# sequential. Profiled boots are NOT scored serving artifacts.
set -uo pipefail
cd /data/smcho/self-spec-moe
PHASE=research/96_selector_foundations
LOG="$PHASE/logs"; mkdir -p "$LOG" "$PHASE/data/w8"

export PATH="/data/smcho/self-spec-moe/.venv/bin:$PATH"
export HF_HOME=/data/smcho/huggingface VLLM_CACHE_ROOT=/data/smcho/.cache/vllm
export TRITON_CACHE_DIR=/data/smcho/.cache/triton TMPDIR=/data/smcho/tmp

COMMON="VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1 VLLM_SELF_SPEC_CPU_ORCH=1"
SHARED="$COMMON VLLM_SELF_SPEC_SHARED_KV=1 VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE=1 VLLM_SELF_SPEC_SKIP_PREFILL_DRAFT=1 VLLM_SELF_SPEC_DRAFT_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1"

MLA_MODEL="deepseek-ai/DeepSeek-V2-Lite"; MLA_DRAFT="/data/smcho/ckpts/DeepSeek-V2-Lite-W8A16-INT8-chan"
MOE_MODEL="Qwen/Qwen3-30B-A3B"; MOE_DRAFT="/data/smcho/ckpts/Qwen3-30B-A3B-W4A16-INT4-sym"

cleanup() {
  local gpus="$1"
  for g in ${gpus//,/ }; do
    for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$g" 2>/dev/null); do
      kill -9 "$p" 2>/dev/null; done
  done
  local w=0
  while [ $w -lt 180 ]; do
    local busy=0
    for g in ${gpus//,/ }; do
      local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null)
      [ "${u:-0}" -gt 2000 ] && busy=1
    done
    [ $busy -eq 0 ] && break
    sleep 10; w=$((w + 10))
  done
  sleep 5
}

boot() {  # gpus model tp draft ceil b k tag timeout
  local gpus="$1" model="$2" tp="$3" draft="$4" ceil="$5" b="$6" k="$7"
  local tag="$8" tmo="$9"
  local dir="$PHASE/data/w8/prof_$tag"
  [ -d "$dir" ] && [ -n "$(ls "$dir" 2>/dev/null)" ] && { echo "[W8] skip $tag"; return 0; }
  mkdir -p "$dir"
  echo "[W8] === $tag gpus=$gpus ==="
  cleanup "$gpus"
  env CUDA_VISIBLE_DEVICES="$gpus" \
      VLLM_SELF_SPEC_PROFILE=1 VLLM_SELF_SPEC_PROFILE_OUT="$dir" \
      G93_MODEL="$model" G93_TP="$tp" G93_DRAFT="$draft" G93_K="$k" \
      G93_BATCHES="$b" G93_DATASETS=R2 G93_ITERS=2 \
      G93_CEILING="$ceil" G93_MAXLEN=24576 G93_TUNE=0 \
      G93_OUT="$dir/serving.json" G93_TAG="w8_$tag" \
      $SHARED \
      timeout "$tmo" .venv/bin/python research/93_c1_grid/scripts/run_grid.py \
        >> "$LOG/w8_$tag.log" 2>&1 || echo "[W8] FAILED $tag"
}

mla_lane() {  # gpu configs...
  local g="$1"; shift
  for cfg in "$@"; do
    local b="${cfg%%:*}" k="${cfg##*:}"
    boot "$g" "$MLA_MODEL" 1 "$MLA_DRAFT" 2048 "$b" "$k" "mla_b${b}_K${k}" 1800
  done
  cleanup "$g"
}

# 6 MLA boots split across lanes (3 each)
mla_lane 0 "1:1" "8:1" "32:1" & P0=$!
mla_lane 1 "1:4" "8:4" "32:4" & P1=$!
wait $P0 $P1
echo "[W8] MLA phase done"

for cfg in "1:1" "1:4" "8:1" "8:4" "32:1" "32:4"; do
  b="${cfg%%:*}"; k="${cfg##*:}"
  boot "0,1" "$MOE_MODEL" 2 "$MOE_DRAFT" 16384 "$b" "$k" "moe_b${b}_K${k}" 3600
done
cleanup "0,1"
echo "[W8] all done: $(ls -d $PHASE/data/w8/prof_* 2>/dev/null | wc -l)/12 dirs"
