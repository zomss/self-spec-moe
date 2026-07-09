#!/bin/bash
# Rank-2 fp8 lever (agent a34fb00c): does forcing the DENSE qkv/o linears to
# weight-only Marlin (VLLM_TEST_FORCE_FP8_MARLIN=1) beat Marlin-MoE-only? On H100
# the dense fp8 linears otherwise fall to cutlass W8A8 (per-token act quant).
# Target is bf16 so this global env only touches the draft's fp8 layers -> safe.
# fp8 draft, moe_backend=marlin, NO window, K=2, 16k b8, clean. Ref: marlin-MoE=606.
set -u
PHASE=/data/smcho/self-spec-moe/research/74_draft_step_cost
P52=/data/smcho/self-spec-moe/research/52_two_node_e2e
P57=/data/smcho/self-spec-moe/research/57_large_ep_spec_strategy
PY=/data/smcho/self-spec-moe/.venv/bin/python
ME="$(whoami)"
kill_mine(){ for p in 'w7_2node[.]py' 'EngineCor[e]' 'Worker_D[P]'; do pkill -9 -u "$ME" -f "$p" 2>/dev/null; done; sleep 5; }
# ARM: moe-only (marlin MoE, dense=cutlass) | dense (marlin MoE + marlin dense)
for ARM in moeonly dense; do
  FORCE=""; [ "$ARM" = dense ] && FORCE="VLLM_TEST_FORCE_FP8_MARLIN=1"
  LOG="$PHASE/logs/wqd_${ARM}_K2_16k_b8.log"
  kill_mine
  ( source "$PHASE/scripts/env_cloud4.sh"
    unset VLLM_SELF_SPEC_COMPILE_CONSISTENT
    export W7_DRAFT_QUANT=fp8 W7_DRAFT_MOE_BACKEND=marlin $FORCE
    export W7_KS=2 W7_BATCHES=8 W7_ITERS=4 W7_WARMUP=1 W7_TAG="p74_wqd_${ARM}_K2_16k_b8" \
      W7_MASTER_PORT=$((17700 + RANDOM%40)) W7_CTX_TOKENS=16384 W7_MAX_MODEL_LEN=20480 \
      W7_MAX_NUM_BATCHED=8192 W7_OUT="$PHASE/data" W7_PROMPT_FILE="$P57/data/prompts_ondist.txt" W7_CHAT=1
    W7_NODE_RANK=0 timeout 1200 "$PY" "$P52/scripts/w7_2node.py" spec ) > "$LOG" 2>&1
  echo "[wqd] ${ARM}: $(grep -hE 'W7-2N.*batch=' "$LOG" | tail -1)"
  grep -hE "Marlin|marlin|FP8 Marlin|Fp8 MoE backend" "$LOG" | tail -3
  kill_mine
done
echo "[wqd] DONE ($(date +%H:%M:%S))"
