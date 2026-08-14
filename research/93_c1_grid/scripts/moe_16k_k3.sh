#!/bin/bash
# Gate-2 step (c): close the MoE crossover question inside the map.
# K3 (the record's winning depth) for the leading arms at all ctxs
# incl. 16k, + K2/K4 16k extensions, + AR@16k. b8x16k = the record's
# winning cell (1.15x fixed-stack, phase 81).
set -uo pipefail
REPO=/data/smcho/self-spec-moe
PHASE=$REPO/research/93_c1_grid
P82DATA=$REPO/research/82_runtime_switching/data
export HF_HOME=/data/smcho/huggingface
export PATH="$REPO/.venv/bin:$PATH"
cd "$REPO"
GPU="6,7"; MODEL="Qwen/Qwen3-30B-A3B"

COMMON="VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1 VLLM_SELF_SPEC_CPU_ORCH=1"
SHARED="$COMMON VLLM_SELF_SPEC_SHARED_KV=1 VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE=1 VLLM_SELF_SPEC_SKIP_PREFILL_DRAFT=1 VLLM_SELF_SPEC_DRAFT_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1"
winplain() { echo "$SHARED VLLM_SELF_SPEC_DRAFT_KV_WINDOW=$1 VLLM_SELF_SPEC_DRAFT_KV_SINKS=16"; }

gpu_cleanup() {
  for i in 6 7; do
    local u; u=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $i)
    for p in $(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | tr -d ',' | awk -v u="$u" '$2==u{print $1}'); do
      [ "$(ps -o user= -p "$p" 2>/dev/null)" = "smcho" ] || continue
      kill "$p" 2>/dev/null; sleep 2; kill -0 "$p" 2>/dev/null && kill -9 "$p" 2>/dev/null
    done
  done
  sleep 3
}

run_one() {  # <lever> <karm> <draft> <ctxs> <extra>
  local lever=$1 karm=$2 draft=$3 ctxs=$4 extra=$5
  local csv="cells_93_moe_${lever}.csv"
  # resume: skip if this K already has a 16000-ctx row (or any row for k3)
  if [ -f "$P82DATA/$csv" ]; then
    if [ "$karm" = "k3" ] && grep -q "^k3," "$P82DATA/$csv"; then
      echo "[C] skip $lever/$karm"; return 0; fi
    if [ "$karm" != "k3" ] && grep -qE "^${karm},[0-9]+,[0-9]+,16000," "$P82DATA/$csv"; then
      echo "[C] skip $lever/$karm@16k"; return 0; fi
  fi
  echo "[C] measure $lever/$karm ctxs=$ctxs"
  env CUDA_VISIBLE_DEVICES=$GPU COMPILE_MODEL="$MODEL" \
      COMPILE_DRAFT="$draft" COMPILE_TP=2 \
      COMPILE_BATCHES="1,4,8,16,32,64,128" COMPILE_CTXS="$ctxs" \
      COMPILE_KV_LIMIT=130000 COMPILE_CELLS="$csv" \
      COMPILE_TABLE="table_93_moe_${lever}.json" \
      $extra timeout 3600 .venv/bin/python \
      "$PHASE/scripts/compile_cells_93.py" --measure "$karm" \
      >> "$PHASE/logs/moe_16k_k3.log" 2>&1 \
      || echo "[C] FAIL $lever/$karm"
  gpu_cleanup
}

ALLCTX="2000,8000,14000,16000"
C16="16000"

# AR baseline at 16k
run_one off off "$MODEL" "$C16" ""

declare -A LEAD
LEAD[w4a16]="/data/smcho/ckpts/Qwen3-30B-A3B-W4A16-INT4-sym|$SHARED VLLM_DISABLED_KERNELS=MacheteLinearKernel"
LEAD[w8chan]="/data/smcho/ckpts/Qwen3-30B-A3B-W8A16-INT8-chan|$SHARED"
LEAD[win512]="$MODEL|$(winplain 512)"
LEAD[win2048]="$MODEL|$(winplain 2048)"
LEAD[win8192]="$MODEL|$(winplain 8192)"
LEAD[skipb2]="$MODEL|$SHARED VLLM_SELF_SPEC_DRAFT_SKIP_LAYERS=15,23"

for lever in "${!LEAD[@]}"; do
  IFS='|' read -r draft extra <<< "${LEAD[$lever]}"
  run_one "$lever" k3 "$draft" "$ALLCTX" "$extra"       # K3 everywhere
  run_one "$lever" k2 "$draft" "$C16" "$extra"          # 16k extension
  run_one "$lever" k4 "$draft" "$C16" "$extra"
done

cp -f "$P82DATA"/cells_93_moe_*.csv "$PHASE/data/" 2>/dev/null
echo "[C] MOE-16K-K3-DONE"
