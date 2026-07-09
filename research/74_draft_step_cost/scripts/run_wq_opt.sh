#!/bin/bash
# Optimize the WEIGHT-QUANT draft: does forcing Marlin fp8 MoE (fast H100, no
# DeepGEMM) beat the auto-picked slow TRITON? fp8 draft, NO window, K=2, 16k b8,
# clean (no profiler, batch-invariant off). Refs: bf16 base K=2 = 592, no-spec = 599.
set -u
PHASE=/data/smcho/self-spec-moe/research/74_draft_step_cost
P52=/data/smcho/self-spec-moe/research/52_two_node_e2e
P57=/data/smcho/self-spec-moe/research/57_large_ep_spec_strategy
PY=/data/smcho/self-spec-moe/.venv/bin/python
ME="$(whoami)"
kill_mine(){ for p in 'w7_2node[.]py' 'EngineCor[e]' 'Worker_D[P]'; do pkill -9 -u "$ME" -f "$p" 2>/dev/null; done; sleep 5; }
for MB in triton marlin; do
  LOG="$PHASE/logs/wq_${MB}_K2_16k_b8.log"
  kill_mine
  ( source "$PHASE/scripts/env_cloud4.sh"
    unset VLLM_SELF_SPEC_COMPILE_CONSISTENT
    export W7_DRAFT_QUANT=fp8 W7_DRAFT_MOE_BACKEND="$MB"
    export W7_KS=2 W7_BATCHES=8 W7_ITERS=4 W7_WARMUP=1 W7_TAG="p74_wq_${MB}_K2_16k_b8" \
      W7_MASTER_PORT=$((17600 + RANDOM%40)) W7_CTX_TOKENS=16384 W7_MAX_MODEL_LEN=20480 \
      W7_MAX_NUM_BATCHED=8192 W7_OUT="$PHASE/data" W7_PROMPT_FILE="$P57/data/prompts_ondist.txt" W7_CHAT=1
    W7_NODE_RANK=0 timeout 1200 "$PY" "$P52/scripts/w7_2node.py" spec ) > "$LOG" 2>&1
  echo "[wq] moe=${MB}: $(grep -hE 'W7-2N.*batch=' "$LOG" | tail -1)"
  grep -hE "Fp8 MoE backend|Marlin|marlin|does not support" "$LOG" | tail -2
  kill_mine
done
echo "[wq] DONE ($(date +%H:%M:%S))"
