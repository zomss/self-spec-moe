#!/usr/bin/env bash
# Phase 48 (shielded sweep): rerun of the Phase-47 sweep2 win curve AFTER fixing
# the delay-shielding gate in vllm/distributed/device_communicators/all2all.py:
# _emulate_exposed_a2a_delay() now returns without sleeping when the comm-free
# local-route flag is active (the self-spec draft), so the draft is no longer
# charged emulated A2A it never performs. The measured curve should now equal
# the (higher) analyzer-model "shielded" curve instead of the conservative
# lower bound. A/B revert knob: W7_CHARGE_DRAFT_A2A=1 reproduces Phase 47.
#
# Everything else is IDENTICAL to research/47_sweep2/scripts/run_sweep2.sh:
# Qwen3-30B-A3B, DP=8/EP=8, forced-PCIe, FP8 full-replica comm-free draft,
# fully-optimized stack, K=2. Harness w7_fp8_timing.py UNCHANGED. NO-SPEC
# baselines REUSED from Phase 42 (no draft -> unaffected by the gate).
# The W7_TAG is kept as sweep2_* so analyze_sweep2.py works via SPEC_DATA/LOGD.
set -u
REPO=/data/smcho/self-spec-moe
cd "$REPO"

export PYTHONPATH="$REPO"
# torch.compile (inductor) needs ninja; the vllm edit invalidates the compile
# cache, so this rerun actually rebuilds. Put the venv's bin on PATH.
export PATH="$REPO/.venv/bin:$PATH"
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
export NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=1
export VLLM_SELF_SPEC_DRAFT_FULL_REPLICA=1
export VLLM_SELF_SPEC_DRAFT_LOCAL_ROUTE=1
export VLLM_SELF_SPEC_DRAFT_FULL_CG=1
export VLLM_SELF_SPEC_COMPILE_CONSISTENT=1
export VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1
# Emit AgRs all2all counts (total/active/real/shielded) at engine destroy for
# the shielding verification (shielded>0 on spec runs, real=0 on the draft).
export VLLM_SELF_SPEC_LOG_A2A_COUNTS=1

export W7_MODEL=/home/smcho/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39
export W7_DP=8 W7_TP=1 W7_TRC=0
export W7_OUT="$REPO/research/48_shielded_sweep/data"
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
LOGD="$REPO/research/48_shielded_sweep/logs"
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
  echo "=== SHIELDED-SWEEP mode=$mode tag=$tag dq=${dq:-bf16} a2a=${a2a}us K=$KVAL dp=$W7_DP piecewise=ON $(date +%T) ==="
  W7_TAG="$tag" W7_DRAFT_QUANT="$dq" W7_A2A_US="$a2a" \
    timeout 6000 "$PY" "$SCRIPT" "$mode" \
    > "$LOGD/${tag}_${mode}_a2a${a2a}.log" 2>&1
  echo "EXIT=$? mode=$mode a2a=$a2a"
  kill_stragglers
}

kill_stragglers
for a2a in $DELAYS; do
  echo "########## A2A delay = ${a2a} us/collective (K=$KVAL, shielded draft) ##########"
  run spec   "sweep2_a2a${a2a}_fp8"  fp8   "$a2a"
done
echo "SHIELDED SWEEP DONE $(date +%T)"
