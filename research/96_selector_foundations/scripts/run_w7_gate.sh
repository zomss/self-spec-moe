#!/usr/bin/env bash
# W7: MLA/MoE Round-2 completeness pass (w7_completeness.md).
# Phase 1: MLA TP1, two parallel single-GPU lanes; lane g boots
#   off then uncond on GPU g (2 certification boots per arm, on
#   different GPU lanes by design -- episode localization).
# Phase 2: MoE TP2 on GPUs 0+1, sequential: off x2, uncond x2.
# All boots: notune (G93_TUNE=0), ITERS=4, batches 1,8,32,64,
# regimes R1,R2,R6, seed-0 content, e3 SHARED env stack.
set -uo pipefail
cd /data/smcho/self-spec-moe
PHASE=research/96_selector_foundations
LOG="$PHASE/logs"; mkdir -p "$LOG" "$PHASE/data/w7"

export PATH="/data/smcho/self-spec-moe/.venv/bin:$PATH"
export HF_HOME=/data/smcho/huggingface VLLM_CACHE_ROOT=/data/smcho/.cache/vllm
export TRITON_CACHE_DIR=/data/smcho/.cache/triton TMPDIR=/data/smcho/tmp

COMMON="VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1 VLLM_SELF_SPEC_CPU_ORCH=1"
SHARED="$COMMON VLLM_SELF_SPEC_SHARED_KV=1 VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE=1 VLLM_SELF_SPEC_SKIP_PREFILL_DRAFT=1 VLLM_SELF_SPEC_DRAFT_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1"

MLA_MODEL="deepseek-ai/DeepSeek-V2-Lite"; MLA_DRAFT="/data/smcho/ckpts/DeepSeek-V2-Lite-W8A16-INT8-chan"
MOE_MODEL="Qwen/Qwen3-30B-A3B"; MOE_DRAFT="/data/smcho/ckpts/Qwen3-30B-A3B-W4A16-INT4-sym"

cleanup() {  # gpus (comma list)
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

boot() {  # gpus model tp draft ceil maxlen tag out timeout extra_env
  local gpus="$1" model="$2" tp="$3" draft="$4" ceil="$5" maxlen="$6"
  local tag="$7" out="$8" tmo="$9" extra="${10}"
  [ -f "$out" ] && { echo "[W7] skip $tag (exists)"; return 0; }
  echo "[W7] === $tag gpus=$gpus ==="
  cleanup "$gpus"
  env CUDA_VISIBLE_DEVICES="$gpus" \
      G93_MODEL="$model" G93_TP="$tp" G93_DRAFT="$draft" G93_K=4 \
      G93_BATCHES=1,8,32,64 G93_DATASETS=R1,R2,R6 G93_ITERS=4 \
      G93_CEILING="$ceil" G93_MAXLEN="$maxlen" G93_TUNE=0 \
      G93_OUT="$out" G93_TAG="$tag" \
      $extra \
      timeout "$tmo" .venv/bin/python research/93_c1_grid/scripts/run_grid.py \
        >> "$LOG/w7_${tag}.log" 2>&1 || echo "[W7] FAILED $tag (see $LOG/w7_${tag}.log)"
}

mla_lane() {  # gpu
  local g="$1"
  boot "$g" "$MLA_MODEL" 1 off         2048 24576 "mla_off_lane$g" \
       "$PHASE/data/w7/w7_mla_off_lane$g.json" 3600 ""
  boot "$g" "$MLA_MODEL" 1 "$MLA_DRAFT" 2048 24576 "mla_uncond_lane$g" \
       "$PHASE/data/w7/w7_mla_uncond_lane$g.json" 3600 "$SHARED"
  cleanup "$g"
}

mla_lane 0 & P0=$!
mla_lane 1 & P1=$!
wait $P0 $P1
echo "[W7] MLA phase complete: $(ls $PHASE/data/w7/w7_mla_*.json 2>/dev/null | wc -l)/4"

for b in 1 2; do
  boot "0,1" "$MOE_MODEL" 2 off          16384 24576 "moe_off_boot$b" \
       "$PHASE/data/w7/w7_moe_off_boot$b.json" 5400 ""
done
for b in 1 2; do
  boot "0,1" "$MOE_MODEL" 2 "$MOE_DRAFT" 16384 24576 "moe_uncond_boot$b" \
       "$PHASE/data/w7/w7_moe_uncond_boot$b.json" 5400 "$SHARED"
done
cleanup "0,1"
echo "[W7] all complete: $(ls $PHASE/data/w7/w7_*.json 2>/dev/null | wc -l)/8"
