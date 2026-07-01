#!/usr/bin/env bash
# Phase 42 COMM SWEEP: Qwen3-30B-A3B, DP=8/EP=8, FP8 full-replica comm-free draft
# vs no-spec, sweeping the EMULATED exposed per-collective A2A delay to find the
# crossover where spec/no-spec speedup passes 1.0x (the comm-bound regime a real
# multi-node all-to-all would create).
#
# Reuses research/34_worldA_system/scripts/w7_fp8_timing.py UNCHANGED (the harness
# that produced Phase 41's 0.45x @ a2a=0 and 0.62x @ a2a=100us, batch=64).
#
# Sweep: W7_A2A_US in {0,100,250,500,1000} us/collective x batch {32,64,128}.
#   - spec  : comm-free full-replica draft skips the AgRs collectives; only the
#             full-EP VERIFY pays the emulated delay.
#   - nospec: plain full-EP decode pays the emulated delay on EVERY collective on
#             EVERY token.
# speedup = spec_tok/s / nospec_tok/s at each (delay, batch).
#
# Forced-PCIe, FULL_CG + COMPILE_CONSISTENT (Phase 41 fixed config), DEEP_GEMM off.
# Serial: ONE engine (one DP group) at a time; kill our OWN stragglers between runs.
set -u
REPO=/data/smcho/self-spec-moe
cd "$REPO"

export PYTHONPATH="$REPO"
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
export NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=1
export VLLM_SELF_SPEC_DRAFT_FULL_CG=1 VLLM_SELF_SPEC_COMPILE_CONSISTENT=1
# Emit AgRs all2all counts (total/active/real) at engine destroy for fairness check.
export VLLM_SELF_SPEC_LOG_A2A_COUNTS=1

export W7_MODEL=/home/smcho/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39
export W7_DP=8 W7_TP=1 W7_TRC=0
export W7_OUT="$REPO/research/42_comm_sweep/data"
export W7_EAGER=0                       # CUDA graphs ON (+ FULL_CG via env above)
export W7_OUTLEN="${W7_OUTLEN:-160}" W7_SHORTLEN="${W7_SHORTLEN:-32}"
export W7_ITERS="${W7_ITERS:-3}" W7_WARMUP="${W7_WARMUP:-2}"
export W7_GPU_MEM="${W7_GPU_MEM:-0.90}"

# K per the sweep spec (K=4). Sanity baseline (Phase 41) was K=3; accept_len is
# ~3.78 at both so no-spec/spec relative timing is comparable.
KVAL="${KVAL:-4}"
export W7_KS="$KVAL"
export W7_BATCHES="${W7_BATCHES:-32,64,128}"
DELAYS="${DELAYS:-0 100 250 500 1000}"

PY="$REPO/.venv/bin/python"
SCRIPT="$REPO/research/34_worldA_system/scripts/w7_fp8_timing.py"
LOGD="$REPO/research/42_comm_sweep/logs"
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
  echo "=== SWEEP mode=$mode tag=$tag dq=${dq:-bf16} a2a=${a2a}us K=$KVAL dp=$W7_DP $(date +%T) ==="
  W7_TAG="$tag" W7_DRAFT_QUANT="$dq" W7_A2A_US="$a2a" \
    timeout 6000 "$PY" "$SCRIPT" "$mode" \
    > "$LOGD/${tag}_${mode}_a2a${a2a}.log" 2>&1
  echo "EXIT=$? mode=$mode a2a=$a2a"
  kill_stragglers
}

kill_stragglers
for a2a in $DELAYS; do
  echo "########## A2A delay = ${a2a} us/collective ##########"
  run nospec "sweep_a2a${a2a}"      ""    "$a2a"
  run spec   "sweep_a2a${a2a}_fp8"  fp8   "$a2a"
done
echo "SWEEP DONE $(date +%T)"
