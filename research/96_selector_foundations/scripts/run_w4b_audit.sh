#!/usr/bin/env bash
# W4b -- llama C2 notune re-audit (w3_preregistration.md section 6.3).
#
# Re-measures the deployed llama config's compile cells under notune:
#   off anchor + q-w4a16_w-{512,2048}_s-b2 x {k2,k4}  = 5 boots,
#   each boot covers the full 9-cell grid (b{1,8,32} x ctx{2k,8k,14k}).
# Same machinery as phase 94's oracle (compile_cells_93.py), COMPILE_TUNE=0.
#
# Pre-registered prediction (results_w2.md F6): audited R ~ table R / 1.46
# UNIFORMLY across (window, K, batch, ctx). If the inflation is not
# uniform, the whole llama table regenerates, not just a scale factor.
set -uo pipefail
REPO=/data/smcho/self-spec-moe
PHASE=$REPO/research/96_selector_foundations
P82DATA=$REPO/research/82_runtime_switching/data
COMPILE=$REPO/research/93_c1_grid/scripts/compile_cells_93.py
cd "$REPO"

export HF_HOME=/data/smcho/huggingface
export PATH="$REPO/.venv/bin:$PATH"
export TRITON_CACHE_DIR=/data/smcho/.cache/triton
export TMPDIR=/data/smcho/tmp
LOG="$PHASE/logs"; mkdir -p "$LOG" "$PHASE/data/w4"

MODEL="NousResearch/Meta-Llama-3.1-8B-Instruct"
DRAFT="$HOME/ckpts/Llama31-8B-Instruct-W4A16-INT4-sym"
SKIPSET="3,8"
COMMON="VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1 VLLM_SELF_SPEC_CPU_ORCH=1"
SHARED="$COMMON VLLM_SELF_SPEC_SHARED_KV=1 VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE=1 VLLM_SELF_SPEC_SKIP_PREFILL_DRAFT=1 VLLM_SELF_SPEC_DRAFT_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1"

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

run_cfg() {  # gpu name draft extra karm
  local gpu="$1" name="$2" draft="$3" extra="$4" karm="$5"
  local csv="w4audit_llama_${name}.csv"
  if [ -f "$P82DATA/$csv" ] && grep -q "^${karm}," "$P82DATA/$csv"; then
    echo "[W4b] skip $name/$karm"; return 0
  fi
  echo "[W4b] === gpu=$gpu $name/$karm ==="
  cleanup "$gpu"
  env CUDA_VISIBLE_DEVICES="$gpu" COMPILE_MODEL="$MODEL" \
      COMPILE_DRAFT="$draft" COMPILE_TP=1 \
      COMPILE_BATCHES="1,8,32" COMPILE_CTXS="2000,8000,14000" \
      COMPILE_KV_LIMIT=260000 COMPILE_CELLS="$csv" \
      COMPILE_TABLE="w4audit_llama_${name}.json" \
      COMPILE_TUNE=0 \
      $extra timeout 3600 .venv/bin/python "$COMPILE" --measure "$karm" \
      >> "$LOG/w4b_${name}.log" 2>&1 \
      || echo "[W4b] FAIL $name/$karm"
  cleanup "$gpu"
}

W512="VLLM_SELF_SPEC_DRAFT_KV_WINDOW=512 VLLM_SELF_SPEC_DRAFT_KV_SINKS=16 VLLM_SELF_SPEC_DRAFT_FULLCG=1"
W2048="VLLM_SELF_SPEC_DRAFT_KV_WINDOW=2048 VLLM_SELF_SPEC_DRAFT_KV_SINKS=16 VLLM_SELF_SPEC_DRAFT_FULLCG=1"
SKIP="VLLM_SELF_SPEC_DRAFT_SKIP_LAYERS=$SKIPSET"

( run_cfg 0 off "$MODEL" "" off
  run_cfg 0 "q-w4a16_w-512_s-b2"  "$DRAFT" "$SHARED $W512 $SKIP"  k2
  run_cfg 0 "q-w4a16_w-512_s-b2"  "$DRAFT" "$SHARED $W512 $SKIP"  k4 ) &
L0=$!
( run_cfg 1 "q-w4a16_w-2048_s-b2" "$DRAFT" "$SHARED $W2048 $SKIP" k2
  run_cfg 1 "q-w4a16_w-2048_s-b2" "$DRAFT" "$SHARED $W2048 $SKIP" k4 ) &
L1=$!
wait $L0 $L1
cp -f "$P82DATA"/w4audit_llama_*.csv "$PHASE/data/w4/" 2>/dev/null
echo "[W4b] audit complete"
ls -la "$PHASE/data/w4/" | grep w4audit