#!/usr/bin/env bash
# W11: llama + q3_32b Stage-B surfaces under EQUAL WORK
# (w11_llama_q32_remeasure.md). Same protocol as W10.
# Phase 1: llama TP1, 6 arms split across two parallel GPU lanes.
# Phase 2: q3_32b TP2, 5 arms sequential on GPUs 0+1.
set -uo pipefail
cd /data/smcho/self-spec-moe
PHASE=research/96_selector_foundations
LOG="$PHASE/logs"; mkdir -p "$LOG" "$PHASE/data/w11"

export PATH="/data/smcho/self-spec-moe/.venv/bin:$PATH"
export HF_HOME=/data/smcho/huggingface VLLM_CACHE_ROOT=/data/smcho/.cache/vllm
export TRITON_CACHE_DIR=/data/smcho/.cache/triton TMPDIR=/data/smcho/tmp

COMMON="VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1 VLLM_SELF_SPEC_CPU_ORCH=1"
SHARED="$COMMON VLLM_SELF_SPEC_SHARED_KV=1 VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE=1 VLLM_SELF_SPEC_SKIP_PREFILL_DRAFT=1 VLLM_SELF_SPEC_DRAFT_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1"
winplain() { echo "$SHARED VLLM_SELF_SPEC_DRAFT_KV_WINDOW=$1 VLLM_SELF_SPEC_DRAFT_KV_SINKS=16"; }

LLAMA_MODEL="NousResearch/Meta-Llama-3.1-8B-Instruct"
LLAMA_FLEN="R1:229,R2:327,R3:218,R4:1032,R5:56,R5cot:655,R6:227,R7:25,R8:737"
Q32_MODEL="Qwen/Qwen3-32B"
Q32_FLEN="R1:326,R2:294,R3:282,R4:983,R5:446,R5cot:1204,R6:102,R7:23,R8:1231"

cleanup() {  # gpus
  for g in ${1//,/ }; do
    for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$g" 2>/dev/null); do
      [ "$(ps -o user= -p "$p" 2>/dev/null)" = "smcho" ] || continue
      kill -9 "$p" 2>/dev/null
    done
  done
  local w=0
  while [ $w -lt 180 ]; do
    local busy=0
    for g in ${1//,/ }; do
      local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null)
      [ "${u:-0}" -gt 2000 ] && busy=1
    done
    [ $busy -eq 0 ] && break
    sleep 10; w=$((w + 10))
  done
  sleep 5
}

boot() {  # gpus arch model tp flen name draft k window extra
  local gpus="$1" arch="$2" model="$3" tp="$4" flen="$5" name="$6"
  local draft="$7" k="$8" window="$9" skip="${10}" extra="${11}"
  local out="$PHASE/data/w11/w11_${arch}_${name}.json"
  if [ -s "$out" ] && grep -q '"complete": true' "$out"; then
    echo "[W11] skip $arch/$name (done)"; return 0
  fi
  echo "[W11] === $arch/$name on gpu $gpus ==="
  cleanup "$gpus"
  env CUDA_VISIBLE_DEVICES="$gpus" \
      G93_MODEL="$model" G93_TP="$tp" G93_DRAFT="$draft" G93_K="$k" \
      G93_WINDOW="$window" G93_SKIP="$skip" \
      G93_BATCHES=1,8,32,64 G93_ITERS=4 \
      G93_CEILING=16384 G93_MAXLEN=24576 \
      G93_TUNE=0 G93_FIXED_LEN="$flen" \
      G93_OUT="$out" G93_TAG="w11_${arch}_${name}" \
      $extra \
      timeout 10800 .venv/bin/python research/93_c1_grid/scripts/run_grid.py \
        >> "$LOG/w11_${arch}_${name}.log" 2>&1 \
        || echo "[W11] FAILED $arch/$name (continuing)"
}

LC="$HOME/ckpts"
llama_lane0() {
  boot 0 llama "$LLAMA_MODEL" 1 "$LLAMA_FLEN" off off 0 0 "" ""
  boot 0 llama "$LLAMA_MODEL" 1 "$LLAMA_FLEN" w4a16_k2 \
       "$LC/Llama31-8B-Instruct-W4A16-INT4-sym" 2 0 "" "$SHARED"
  boot 0 llama "$LLAMA_MODEL" 1 "$LLAMA_FLEN" w8int8_k2 \
       "$LC/Llama31-8B-Instruct-W8A16-INT8-sym" 2 0 "" "$SHARED"
  cleanup 0
}
llama_lane1() {
  boot 1 llama "$LLAMA_MODEL" 1 "$LLAMA_FLEN" w4a16_k4 \
       "$LC/Llama31-8B-Instruct-W4A16-INT4-sym" 4 0 "" "$SHARED"
  boot 1 llama "$LLAMA_MODEL" 1 "$LLAMA_FLEN" win512_k2 \
       self 2 512 "" "$(winplain 512)"
  boot 1 llama "$LLAMA_MODEL" 1 "$LLAMA_FLEN" win2048_k4 \
       self 4 2048 "" "$(winplain 2048)"
  cleanup 1
}

llama_lane0 & P0=$!
llama_lane1 & P1=$!
wait $P0 $P1
echo "[W11] llama done: $(ls $PHASE/data/w11/w11_llama_*.json 2>/dev/null | wc -l)/6"

boot 0,1 q3_32b "$Q32_MODEL" 2 "$Q32_FLEN" off off 0 0 "" ""
boot 0,1 q3_32b "$Q32_MODEL" 2 "$Q32_FLEN" w4gptq_k4 \
     "$LC/Qwen3-32B-W4A16-INT4-gptq" 4 0 "" "$SHARED"
boot 0,1 q3_32b "$Q32_MODEL" 2 "$Q32_FLEN" w4gptq_k6 \
     "$LC/Qwen3-32B-W4A16-INT4-gptq" 6 0 "" "$SHARED"
boot 0,1 q3_32b "$Q32_MODEL" 2 "$Q32_FLEN" win512_k4 \
     self 4 512 "" "$(winplain 512)"
boot 0,1 q3_32b "$Q32_MODEL" 2 "$Q32_FLEN" skipb2_k4 \
     self 4 0 "7,16" "$SHARED"
cleanup 0,1
echo "[W11] done: llama $(ls $PHASE/data/w11/w11_llama_*.json 2>/dev/null | wc -l)/6, q3_32b $(ls $PHASE/data/w11/w11_q3_32b_*.json 2>/dev/null | wc -l)/5"
