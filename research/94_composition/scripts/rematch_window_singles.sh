#!/bin/bash
# Phase 94 control: re-measure C1's WINDOW SINGLES on the FULLCG
# scratchpad chain, so window singles and window-containing
# compositions share one realization.
#
# Why: C1 measured windows with winplain (no FULLCG) -- an IMA-era
# conservative choice. Phase-94 compositions use FULLCG (the IMA was
# localized to wholechain and fixed). FULLCG REQUIRES a window (engine
# constraint), so quant/skip singles cannot have it -- but window
# singles can and should. Without this control, "composition gain"
# partly measures the chain realization, not the composition.
set -uo pipefail
ARCH=${1:?usage: rematch_window_singles.sh dense|llama|q3_32b}
REPO=/data/smcho/self-spec-moe
PHASE=$REPO/research/94_composition
P82DATA=$REPO/research/82_runtime_switching/data
COMPILE=$REPO/research/93_c1_grid/scripts/compile_cells_93.py
export HF_HOME=/data/smcho/huggingface
export PATH="$REPO/.venv/bin:$PATH"
cd "$REPO"

COMMON="VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1 VLLM_SELF_SPEC_CPU_ORCH=1"
SHARED="$COMMON VLLM_SELF_SPEC_SHARED_KV=1 VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE=1 VLLM_SELF_SPEC_SKIP_PREFILL_DRAFT=1 VLLM_SELF_SPEC_DRAFT_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1"
win() { echo "VLLM_SELF_SPEC_DRAFT_KV_WINDOW=$1 VLLM_SELF_SPEC_DRAFT_KV_SINKS=16 VLLM_SELF_SPEC_DRAFT_FULLCG=1"; }

case "$ARCH" in
  dense)  MODEL="Qwen/Qwen3-8B"; TP=1; KVLIM=260000; GPU="${RM_GPU:-6}"; KS="k2 k4";;
  llama)  MODEL="NousResearch/Meta-Llama-3.1-8B-Instruct"; TP=1; KVLIM=260000; GPU="${RM_GPU:-7}"; KS="k2 k4";;
  q3_32b) MODEL="Qwen/Qwen3-32B"; TP=2; KVLIM=130000; GPU="${RM_GPU:-6,7}"; KS="k4";;
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

for W in 512 2048; do
  csv="cells_94ctl_${ARCH}_win${W}fullcg.csv"
  for karm in $KS; do
    if [ -f "$P82DATA/$csv" ] && grep -q "^${karm}," "$P82DATA/$csv"; then
      echo "[CTL:$ARCH] skip win${W}/$karm"; continue
    fi
    echo "[CTL:$ARCH] measure win${W}-FULLCG/$karm"
    env CUDA_VISIBLE_DEVICES=$GPU COMPILE_MODEL="$MODEL" \
        COMPILE_DRAFT="$MODEL" COMPILE_TP=$TP \
        COMPILE_BATCHES="1,8,32" COMPILE_CTXS="2000,8000,14000" \
        COMPILE_KV_LIMIT=$KVLIM COMPILE_CELLS="$csv" \
        COMPILE_TABLE="table_94ctl_${ARCH}_win${W}.json" \
        $SHARED $(win $W) timeout 3600 .venv/bin/python "$COMPILE" --measure "$karm" \
        >> "$PHASE/logs/ctl_${ARCH}.log" 2>&1 \
        || echo "[CTL:$ARCH] FAIL win${W}/$karm"
    gpu_cleanup
  done
done
cp -f "$P82DATA"/cells_94ctl_${ARCH}_*.csv "$PHASE/data/" 2>/dev/null
echo "[CTL:$ARCH] CONTROL-${ARCH}-DONE"
