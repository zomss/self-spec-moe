#!/usr/bin/env bash
# Diagnose the llama R8 gate failure (S=0.935 vs AR, policy armed ~100%).
#
# Predicted-if-always-armed S = 0.931 (live tau 2.296 -> f 0.648, map R 0.733)
# vs measured 0.935: the cost model is right, the policy simply never
# disarmed. Probe duty (3.1% -> S 0.998) and the b8/b32 cell tie (both say
# disarm) are ruled out arithmetically. Remaining hypothesis H1: b16 sits
# EXACTLY on the batch-band boundary (15 -> bit_length 4, 16 -> 5), and a band
# change clears every per-request accept-EMA back to optimistic 1.0
# (scheduler.py:1306-1308), which re-arms at S=1.217.
#
# VLLM_SELF_SPEC_GATE_DEBUG=1 emits per step:
#   [kpick] step=N n_run=N cell=bB/cC f=F K=K
# so n_run (hence the band), the live f, and the chosen K are all observable.
set -uo pipefail
cd /data/smcho/self-spec-moe

GPU="${E95_GPU:-1}"
PHASE=research/95_c3_deploy
LOG="$PHASE/logs"; mkdir -p "$LOG"

export PATH="/data/smcho/self-spec-moe/.venv/bin:$PATH"
export HF_HOME=/data/smcho/huggingface
export VLLM_CACHE_ROOT=/data/smcho/.cache/vllm
export TRITON_CACHE_DIR=/data/smcho/.cache/triton
export TMPDIR=/data/smcho/tmp
export CUDA_VISIBLE_DEVICES="$GPU"

export VLLM_SELF_SPEC_DRAFT_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1
export VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1
export VLLM_SELF_SPEC_SHARED_KV=1 VLLM_SELF_SPEC_DRAFT_KV_SINKS=16
export VLLM_SELF_SPEC_CPU_ORCH=1 VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE=1
export VLLM_SELF_SPEC_SKIP_PREFILL_DRAFT=1
export VLLM_SELF_SPEC_DRAFT_FULLCG=1 VLLM_SELF_SPEC_DRAFT_STEP0_FULL_CG=1
export VLLM_SELF_SPEC_DRAFT_WHOLECHAIN=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_SELF_SPEC_GATE_DEBUG=1          # <- the diagnostic

for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$GPU" 2>/dev/null); do
  kill -9 "$p" 2>/dev/null
done
sleep 5

E95_ARCH=llama E95_WINDOW=512 E95_SEED=0 E95_REGIMES=R8 E95_ITERS=1 \
  E95_OUT="$PHASE/data/diag_r8_llama_w512.json" \
  timeout 1800 .venv/bin/python "$PHASE/scripts/run_e0_envelope.py" \
    > "$LOG/diag_r8.log" 2>&1
echo "[diag] rc=$? -> $LOG/diag_r8.log"
grep -c "\[kpick\]" "$LOG/diag_r8.log" | sed 's/^/[diag] kpick lines: /'
