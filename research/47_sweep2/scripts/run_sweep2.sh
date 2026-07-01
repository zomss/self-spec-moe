#!/usr/bin/env bash
# Phase 47 (sweep2): definitive speedup-vs-A2A win curve for the fully-optimized
# comm-free self-spec (genuine replica + PIECEWISE draft chain + all fixes), K=2.
#
# Model Qwen3-30B-A3B, DP=8/EP=8, forced-PCIe, FP8 full-replica comm-free draft.
# Reuses research/34_worldA_system/scripts/w7_fp8_timing.py UNCHANGED (same harness
# and methodology as Phase 42; the only deltas vs Phase 42 are: PIECEWISE draft
# chain ON, and K=2 instead of K=4).
#
# NO-SPEC baselines are REUSED from Phase 42 (research/42_comm_sweep/data), which are
# unchanged by piecewise (piecewise only affects the spec draft chain). This driver
# runs ONLY the spec config; the analyzer pulls Phase-42 no-spec for the ratio.
#
# Sweep: W7_A2A_US in {0,100,250,500,1000} us/collective x batch {32,64,128}.
# speedup = spec tok/s / (Phase-42 no-spec tok/s) at each (delay, batch).
#
# Hygiene: DEEP_GEMM off; serial single-engine; tear down OWN workers after each run.
set -u
REPO=/data/smcho/self-spec-moe
cd "$REPO"

export PYTHONPATH="$REPO"
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
export NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=1
# FULLY-OPTIMIZED comm-free self-spec: genuine full-replica draft + local route +
# full CG + compile-consistent + PIECEWISE draft chain (the Phase 43-46 stack).
export VLLM_SELF_SPEC_DRAFT_FULL_REPLICA=1
export VLLM_SELF_SPEC_DRAFT_LOCAL_ROUTE=1
export VLLM_SELF_SPEC_DRAFT_FULL_CG=1
export VLLM_SELF_SPEC_COMPILE_CONSISTENT=1
export VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1
# Emit AgRs all2all counts (total/active/real) at engine destroy for the
# draft-shielding fairness check (Report item 4).
export VLLM_SELF_SPEC_LOG_A2A_COUNTS=1

export W7_MODEL=/home/smcho/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39
export W7_DP=8 W7_TP=1 W7_TRC=0
export W7_OUT="$REPO/research/47_sweep2/data"
export W7_EAGER=0                       # CUDA graphs ON (+ FULL_CG + PIECEWISE)
export W7_OUTLEN="${W7_OUTLEN:-160}" W7_SHORTLEN="${W7_SHORTLEN:-32}"
export W7_ITERS="${W7_ITERS:-3}" W7_WARMUP="${W7_WARMUP:-2}"
export W7_GPU_MEM="${W7_GPU_MEM:-0.90}"

KVAL="${KVAL:-2}"
export W7_KS="$KVAL"
export W7_BATCHES="${W7_BATCHES:-32,64,128}"
DELAYS="${DELAYS:-0 100 250 500 1000}"

PY="$REPO/.venv/bin/python"
SCRIPT="$REPO/research/34_worldA_system/scripts/w7_fp8_timing.py"
LOGD="$REPO/research/47_sweep2/logs"
mkdir -p "$LOGD" "$W7_OUT"

kill_stragglers() {
  pkill -9 -f "w7_fp8_timing" 2>/dev/null
  for pid in $(ps -u "$USER" -o pid,cmd | grep -E "EngineCore|VLLM::" | grep -v grep | awk '{print $1}'); do
    kill -9 "$pid" 2>/dev/null
  done
  sleep 5
}

run() {  # mode tag draft_quant a2a_us
  local mode="$1" tag="$2" dq="${3:-}" a2a="$4"
  echo "=== SWEEP2 mode=$mode tag=$tag dq=${dq:-bf16} a2a=${a2a}us K=$KVAL dp=$W7_DP piecewise=ON $(date +%T) ==="
  W7_TAG="$tag" W7_DRAFT_QUANT="$dq" W7_A2A_US="$a2a" \
    timeout 6000 "$PY" "$SCRIPT" "$mode" \
    > "$LOGD/${tag}_${mode}_a2a${a2a}.log" 2>&1
  echo "EXIT=$? mode=$mode a2a=$a2a"
  kill_stragglers
}

kill_stragglers
for a2a in $DELAYS; do
  echo "########## A2A delay = ${a2a} us/collective (K=$KVAL, piecewise ON) ##########"
  run spec   "sweep2_a2a${a2a}_fp8"  fp8   "$a2a"
done
echo "SWEEP2 DONE $(date +%T)"
