#!/bin/bash
# Phase 94 Step 1: composition premise probe (P1/P2).
# Measures COMPOSED configs on the C1 compile-cell protocol so they are
# directly comparable to C1's single-lever cells (no singles re-measure).
#
# Usage: run_premise_probe.sh {dense|llama|q3_32b}
# Env: PROBE_GPU
set -uo pipefail
ARCH=${1:?usage: run_premise_probe.sh dense|llama|q3_32b}
REPO=/data/smcho/self-spec-moe
PHASE=$REPO/research/94_composition
P82DATA=$REPO/research/82_runtime_switching/data
COMPILE=$REPO/research/93_c1_grid/scripts/compile_cells_93.py
export HF_HOME=/data/smcho/huggingface
export PATH="$REPO/.venv/bin:$PATH"
mkdir -p "$PHASE/logs" "$PHASE/data"
cd "$REPO"

COMMON="VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1 VLLM_SELF_SPEC_CPU_ORCH=1"
SHARED="$COMMON VLLM_SELF_SPEC_SHARED_KV=1 VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE=1 VLLM_SELF_SPEC_SKIP_PREFILL_DRAFT=1 VLLM_SELF_SPEC_DRAFT_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1"
win() { echo "VLLM_SELF_SPEC_DRAFT_KV_WINDOW=$1 VLLM_SELF_SPEC_DRAFT_KV_SINKS=16 VLLM_SELF_SPEC_DRAFT_FULLCG=1"; }

# comp spec: name|draft|extra-envs   (K grid appended per arch)
declare -a COMPS
case "$ARCH" in
  dense)
    MODEL="Qwen/Qwen3-8B"; TP=1; KVLIM=260000; GPU="${PROBE_GPU:-6}"
    BATCHES="1,8,32"; CTXS="2000,8000,14000"; KS="k2 k4"
    Q4A8=/data/smcho/ckpts/Qwen3-8B-W4A8-gptq
    Q4A16=/data/smcho/ckpts/Qwen3-8B-W4A16-INT4
    HUM="VLLM_DISABLED_KERNELS=MacheteLinearKernel,CutlassW4A8LinearKernel,AllSparkLinearKernel"
    SK="VLLM_SELF_SPEC_DRAFT_SKIP_LAYERS=2,8"
    COMPS=(
      "hum_x_win512|$Q4A8|$SHARED $HUM $(win 512)"
      "hum_x_win2048|$Q4A8|$SHARED $HUM $(win 2048)"
      "hum_x_skipb2|$Q4A8|$SHARED $HUM $SK"
      "hum_x_win512_x_skipb2|$Q4A8|$SHARED $HUM $(win 512) $SK"
      "w4a16_x_win512|$Q4A16|$SHARED $(win 512)"
      "win512_x_skipb2|$MODEL|$SHARED $(win 512) $SK"
    )
    ;;
  llama)
    MODEL="NousResearch/Meta-Llama-3.1-8B-Instruct"; TP=1; KVLIM=260000
    GPU="${PROBE_GPU:-7}"; BATCHES="1,8,32"; CTXS="2000,8000,14000"; KS="k2 k4"
    Q4=/data/smcho/ckpts/Llama31-8B-Instruct-W4A16-INT4-sym
    SK="VLLM_SELF_SPEC_DRAFT_SKIP_LAYERS=3,8"
    COMPS=(
      "w4a16_x_win512|$Q4|$SHARED $(win 512)"
      "w4a16_x_win2048|$Q4|$SHARED $(win 2048)"
      "w4a16_x_skipb2|$Q4|$SHARED $SK"
      "w4a16_x_win512_x_skipb2|$Q4|$SHARED $(win 512) $SK"
    )
    ;;
  q3_32b)
    MODEL="Qwen/Qwen3-32B"; TP=2; KVLIM=130000; GPU="${PROBE_GPU:-6,7}"
    BATCHES="1,8,32"; CTXS="2000,8000,14000"; KS="k4"
    # Machete (W4-GPTQ): Humming wedges at TP2 odd width (C1 ledger)
    Q4G=/data/smcho/ckpts/Qwen3-32B-W4A16-INT4-gptq
    SK="VLLM_SELF_SPEC_DRAFT_SKIP_LAYERS=7,16"
    COMPS=(
      "w4gptq_x_win512|$Q4G|$SHARED $(win 512)"
      "w4gptq_x_win2048|$Q4G|$SHARED $(win 2048)"
      "w4gptq_x_skipb2|$Q4G|$SHARED $SK"
      "w4gptq_x_win512_x_skipb2|$Q4G|$SHARED $(win 512) $SK"
      "win512_x_skipb2|$MODEL|$SHARED $(win 512) $SK"
    )
    ;;
  *) echo "unknown arch $ARCH"; exit 1;;
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

for spec in "${COMPS[@]}"; do
  IFS='|' read -r name draft extra <<< "$spec"
  csv="cells_94_${ARCH}_${name}.csv"
  for karm in $KS; do
    if [ -f "$P82DATA/$csv" ] && grep -q "^${karm}," "$P82DATA/$csv"; then
      echo "[P:$ARCH] skip $name/$karm (done)"; continue
    fi
    echo "[P:$ARCH] measure $name/$karm"
    env CUDA_VISIBLE_DEVICES=$GPU COMPILE_MODEL="$MODEL" \
        COMPILE_DRAFT="$draft" COMPILE_TP=$TP \
        COMPILE_BATCHES="$BATCHES" COMPILE_CTXS="$CTXS" \
        COMPILE_KV_LIMIT=$KVLIM COMPILE_CELLS="$csv" \
        COMPILE_TABLE="table_94_${ARCH}_${name}.json" \
        $extra timeout 3600 .venv/bin/python "$COMPILE" --measure "$karm" \
        >> "$PHASE/logs/probe_${ARCH}.log" 2>&1 \
        || echo "[P:$ARCH] FAIL $name/$karm (continuing)"
    gpu_cleanup
  done
done
cp -f "$P82DATA"/cells_94_${ARCH}_*.csv "$PHASE/data/" 2>/dev/null
echo "[P:$ARCH] PROBE-${ARCH}-DONE"
