#!/bin/bash
# A/B the window-compaction fix. Identical stack, identical prompts; the ONLY
# difference is W7_WIN_FULL_GATHER, which restores the pre-fix gather over the
# full block-table width.
#
# Two things to check:
#   acceptance MUST be identical  -- the narrowing is provably equivalent, so
#                                    any tau difference means a real bug
#   throughput SHOULD improve     -- most at narrow window x long context,
#                                    where the wasted width ratio is largest
set -uo pipefail
REPO=/data/smcho/self-spec-moe; PHASE=$REPO/research/93_c1_grid
P82DATA=$REPO/research/82_runtime_switching/data
COMPILE=$PHASE/scripts/compile_cells_93.py
export HF_HOME=/data/smcho/huggingface; export PATH="$REPO/.venv/bin:$PATH"; cd "$REPO"
GPU="${AB_GPU:-0}"; MODEL="Qwen/Qwen3-8B"; KVLIM=260000
BATCHES="1,4,8,16,32,64,128"; CTXS="2000,8000,14000"
COMMON="VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1 VLLM_SELF_SPEC_CPU_ORCH=1"
SHARED="$COMMON VLLM_SELF_SPEC_SHARED_KV=1 VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE=1 VLLM_SELF_SPEC_SKIP_PREFILL_DRAFT=1 VLLM_SELF_SPEC_DRAFT_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1"
gpu_cleanup() {
  local u; u=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$GPU")
  for p in $(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | tr -d ',' | awk -v u="$u" '$2==u{print $1}'); do
    [ "$(ps -o user= -p "$p" 2>/dev/null)" = "smcho" ] || continue
    kill "$p" 2>/dev/null; sleep 2; kill -0 "$p" 2>/dev/null && kill -9 "$p" 2>/dev/null
  done; sleep 3
}
for WIN in ${AB_WINS:-128 512 2048}; do
  for VARIANT in ${AB_VARIANTS:-old new}; do
    EXTRA=""; case "$VARIANT" in old|old2) EXTRA="W7_WIN_FULL_GATHER=1";; esac
    csv="ab_win${WIN}_${VARIANT}.csv"
    [ -f "$P82DATA/$csv" ] && grep -q '^k2,' "$P82DATA/$csv" && { echo "[AB] skip win$WIN/$VARIANT"; continue; }
    echo "[AB] measure win$WIN / $VARIANT"
    env CUDA_VISIBLE_DEVICES=$GPU COMPILE_MODEL="$MODEL" COMPILE_DRAFT="$MODEL" \
        COMPILE_TP=1 COMPILE_BATCHES="$BATCHES" COMPILE_CTXS="$CTXS" \
        COMPILE_KV_LIMIT=$KVLIM COMPILE_CELLS="$csv" \
        COMPILE_TABLE="ab_win${WIN}_${VARIANT}.json" \
        VLLM_SELF_SPEC_DRAFT_KV_WINDOW=$WIN VLLM_SELF_SPEC_DRAFT_KV_SINKS=16 \
        VLLM_SELF_SPEC_DRAFT_FULLCG=1 \
        $SHARED $EXTRA timeout 5400 .venv/bin/python "$COMPILE" --measure k2 \
        >> "$PHASE/logs/window_ab.log" 2>&1 || echo "[AB] FAIL win$WIN/$VARIANT"
    gpu_cleanup
  done
done
cp -f "$P82DATA"/ab_win*.csv "$PHASE/data/" 2>/dev/null
echo "[AB] WINDOW-AB-DONE"
