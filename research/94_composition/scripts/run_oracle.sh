#!/bin/bash
# Phase 94 Step 2: the ORACLE — exhaustive factorial over a reduced but
# COMPLETE composed space, so any search claim has a ground truth.
#
# Space (per arch): quant{none,q1,q2} x window{none,512,2048}
#                   x skip{none,b2} x K{2,4}  = 36 configs
# Each boot measures the whole batch x ctx grid (b{1,8,32} x
# ctx{2k,8k,14k} = 9 cells), so 36 boots cover 9 cells exhaustively.
#
# Realization: window-containing configs use FULLCG (only chain the
# engine allows with a window; measured worth -0.5% mean, so it does
# not drive the comparison). Everything else plain. Recorded per row.
#
# Usage: run_oracle.sh {dense|llama} ; Env: ORACLE_GPU
set -uo pipefail
ARCH=${1:?usage: run_oracle.sh dense|llama}
REPO=/data/smcho/self-spec-moe
PHASE=$REPO/research/94_composition
P82DATA=$REPO/research/82_runtime_switching/data
COMPILE=$REPO/research/93_c1_grid/scripts/compile_cells_93.py
export HF_HOME=/data/smcho/huggingface
export PATH="$REPO/.venv/bin:$PATH"
cd "$REPO"

COMMON="VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1 VLLM_SELF_SPEC_CPU_ORCH=1"
SHARED="$COMMON VLLM_SELF_SPEC_SHARED_KV=1 VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE=1 VLLM_SELF_SPEC_SKIP_PREFILL_DRAFT=1 VLLM_SELF_SPEC_DRAFT_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1"

case "$ARCH" in
  dense)
    MODEL="Qwen/Qwen3-8B"; TP=1; KVLIM=260000; GPU="${ORACLE_GPU:-6}"
    Q1_NAME=hum;   Q1_CKPT=$HOME/ckpts/Qwen3-8B-W4A8-gptq
    Q1_EXTRA="VLLM_DISABLED_KERNELS=MacheteLinearKernel,CutlassW4A8LinearKernel,AllSparkLinearKernel"
    Q2_NAME=w4a16; Q2_CKPT=$HOME/ckpts/Qwen3-8B-W4A16-INT4; Q2_EXTRA=""
    SKIPSET="2,8"
    ;;
  llama)
    MODEL="NousResearch/Meta-Llama-3.1-8B-Instruct"; TP=1; KVLIM=260000
    GPU="${ORACLE_GPU:-7}"
    Q1_NAME=w4a16; Q1_CKPT=$HOME/ckpts/Llama31-8B-Instruct-W4A16-INT4-sym; Q1_EXTRA=""
    Q2_NAME=w8int8; Q2_CKPT=$HOME/ckpts/Llama31-8B-Instruct-W8A16-INT8-sym; Q2_EXTRA=""
    SKIPSET="3,8"
    ;;
  *) echo "unknown arch"; exit 1;;
esac

gpu_cleanup() {
  for i in ${GPU//,/ }; do
    local u; u=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$i")
    for p in $(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | tr -d ',' | awk -v u="$u" '$2==u{print $1}'); do
      [ "$(ps -o user= -p "$p" 2>/dev/null)" = "smcho" ] || continue
      kill "$p" 2>/dev/null; sleep 2; kill -0 "$p" 2>/dev/null && kill -9 "$p" 2>/dev/null
    done
  done
  sleep 3
}

run_cfg() {  # <name> <draft> <extra> <karm>
  local name=$1 draft=$2 extra=$3 karm=$4
  local csv="oracle_${ARCH}_${name}.csv"
  if [ -f "$P82DATA/$csv" ] && grep -q "^${karm}," "$P82DATA/$csv"; then
    echo "[O:$ARCH] skip $name/$karm"; return 0
  fi
  echo "[O:$ARCH] measure $name/$karm"
  env CUDA_VISIBLE_DEVICES=$GPU COMPILE_MODEL="$MODEL" \
      COMPILE_DRAFT="$draft" COMPILE_TP=$TP \
      COMPILE_BATCHES="1,8,32" COMPILE_CTXS="2000,8000,14000" \
      COMPILE_KV_LIMIT=$KVLIM COMPILE_CELLS="$csv" \
      COMPILE_TABLE="oracle_${ARCH}_${name}.json" \
      $extra timeout 3600 .venv/bin/python "$COMPILE" --measure "$karm" \
      >> "$PHASE/logs/oracle_${ARCH}.log" 2>&1 \
      || echo "[O:$ARCH] FAIL $name/$karm"
  gpu_cleanup
}

# full factorial: quant{none,Q1,Q2} x window{none,512,2048} x skip{none,b2}
for q in none $Q1_NAME $Q2_NAME; do
  case $q in
    none)      qd="$MODEL"; qe="";;
    $Q1_NAME)  qd="$Q1_CKPT"; qe="$Q1_EXTRA";;
    $Q2_NAME)  qd="$Q2_CKPT"; qe="$Q2_EXTRA";;
  esac
  for w in none 512 2048; do
    if [ "$w" = none ]; then we=""; else
      we="VLLM_SELF_SPEC_DRAFT_KV_WINDOW=$w VLLM_SELF_SPEC_DRAFT_KV_SINKS=16 VLLM_SELF_SPEC_DRAFT_FULLCG=1"
    fi
    for s in none b2; do
      if [ "$s" = none ]; then se=""; else
        se="VLLM_SELF_SPEC_DRAFT_SKIP_LAYERS=$SKIPSET"
      fi
      name="q-${q}_w-${w}_s-${s}"
      for karm in k2 k4; do
        run_cfg "$name" "$qd" "$SHARED $qe $we $se" "$karm"
      done
    done
  done
done
cp -f "$P82DATA"/oracle_${ARCH}_*.csv "$PHASE/data/" 2>/dev/null
echo "[O:$ARCH] ORACLE-${ARCH}-DONE"
