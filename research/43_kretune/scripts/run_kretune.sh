#!/usr/bin/env bash
# Phase 43 K-RETUNE: Qwen3-30B-A3B, DP=8/EP=8, FP8 full-replica comm-free draft.
# Measures K in {2,3} to find the best-K speedup curve vs emulated per-collective
# A2A delay. Phase 42 found K=4 never crosses 1.0x <=1000us (plateau ~0.82x) and was
# WORSE than K=3; this retune measures whether a smaller K (fewer draft forwards ->
# cheaper cycle, lower accept_len) crosses 1.0x at a realistic A2A cost.
#
# REUSES research/34_worldA_system/scripts/w7_fp8_timing.py UNCHANGED and the Phase-42
# no-spec baselines (research/42_comm_sweep/data, K-independent) -> spec runs only.
#
# Sweep: W7_KS in {2,3} x W7_A2A_US in {0,100,250,500} us/collective, batch 64
#   (batches configurable via W7_BATCHES; add 32,128 only if time permits).
#   spec: comm-free full-replica draft; only the full-EP VERIFY pays a REAL collective
#   (but the draft ALSO eats the injected sleep -> over-charge; see shielded analysis).
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
export W7_OUT="$REPO/research/43_kretune/data"
export W7_EAGER=0                       # CUDA graphs ON (+ FULL_CG via env above)
export W7_OUTLEN="${W7_OUTLEN:-160}" W7_SHORTLEN="${W7_SHORTLEN:-32}"
export W7_ITERS="${W7_ITERS:-3}" W7_WARMUP="${W7_WARMUP:-2}"
export W7_GPU_MEM="${W7_GPU_MEM:-0.90}"

export W7_BATCHES="${W7_BATCHES:-64}"
KS="${KS:-2 3}"
DELAYS="${DELAYS:-0 100 250 500}"

PY="$REPO/.venv/bin/python"
SCRIPT="$REPO/research/34_worldA_system/scripts/w7_fp8_timing.py"
LOGD="$REPO/research/43_kretune/logs"
mkdir -p "$LOGD" "$W7_OUT"

kill_stragglers() {
  pkill -9 -f "w7_fp8_timing" 2>/dev/null
  for pid in $(ps -u "$USER" -o pid,cmd | grep -E "EngineCore|VLLM::" | grep -v grep | awk '{print $1}'); do
    kill -9 "$pid" 2>/dev/null
  done
  sleep 5
}

run_spec() {  # k a2a_us
  local k="$1" a2a="$2"
  local tag="kr_a2a${a2a}_fp8"
  echo "=== KRETUNE spec K=$k a2a=${a2a}us dp=$W7_DP batches=$W7_BATCHES $(date +%T) ==="
  W7_TAG="$tag" W7_DRAFT_QUANT="fp8" W7_A2A_US="$a2a" W7_KS="$k" \
    timeout 6000 "$PY" "$SCRIPT" spec \
    > "$LOGD/${tag}_spec_a2a${a2a}_K${k}.log" 2>&1
  echo "EXIT=$? K=$k a2a=$a2a"
  kill_stragglers
}

kill_stragglers
for k in $KS; do
  echo "########## K = $k ##########"
  for a2a in $DELAYS; do
    run_spec "$k" "$a2a"
  done
done
echo "KRETUNE DONE $(date +%T)"
