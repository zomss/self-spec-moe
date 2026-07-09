#!/bin/bash
# LAST open quant question: can weight-quant win at HIGH BATCH (weight-read becomes
# binding)? b64, short 2k ctx (KV small -> compute/weight-read bound, NOT KV-bound).
# Q1: does self-spec even help at high batch? (spec-bf16 vs nospec)
# Q2: does draft-only Marlin fp8 move the ratio there? (spec-fp8marlin vs spec-bf16)
# Prediction: high batch -> verify processes B*(K+1) tokens -> compute-heavy verify
# -> self-spec loses; and Marlin dequants fp8->bf16 so it won't help (native fp8 MoE
# unavailable on H100-no-DeepGEMM). This CONFIRMS-or-REFUTES with data. clean.
set -u
PHASE=/data/smcho/self-spec-moe/research/74_draft_step_cost
P52=/data/smcho/self-spec-moe/research/52_two_node_e2e
P57=/data/smcho/self-spec-moe/research/57_large_ep_spec_strategy
PY=/data/smcho/self-spec-moe/.venv/bin/python
ME="$(whoami)"
kill_mine(){ for p in 'w7_2node[.]py' 'EngineCor[e]' 'Worker_D[P]'; do pkill -9 -u "$ME" -f "$p" 2>/dev/null; done; sleep 5; }
run(){  # tag mode quant moe_backend K
  local TAG=$1 MODE=$2 Q=$3 MB=$4 K=$5
  local LOG="$PHASE/logs/hb_${TAG}.log"
  local QENV=""; [ -n "$Q" ] && QENV="W7_DRAFT_QUANT=$Q"
  local MBENV=""; [ -n "$MB" ] && MBENV="W7_DRAFT_MOE_BACKEND=$MB"
  kill_mine
  ( source "$PHASE/scripts/env_cloud4.sh"
    unset VLLM_SELF_SPEC_COMPILE_CONSISTENT
    export $QENV $MBENV
    export W7_KS=$K W7_BATCHES=64 W7_ITERS=4 W7_WARMUP=1 W7_TAG="p74_hb_${TAG}" \
      W7_MASTER_PORT=$((18000 + K + RANDOM%30)) W7_CTX_TOKENS=2048 W7_MAX_MODEL_LEN=4096 \
      W7_MAX_NUM_BATCHED=8192 W7_OUT="$PHASE/data" W7_PROMPT_FILE="$P57/data/prompts_ondist.txt" W7_CHAT=1
    W7_NODE_RANK=0 timeout 1400 "$PY" "$P52/scripts/w7_2node.py" "$MODE" ) > "$LOG" 2>&1
  echo "[hb] ${TAG}: $(grep -hE 'W7-2N.*batch=' "$LOG" | tail -1)"
  kill_mine
}
run nospec        nospec ""    ""      1
run spec_bf16_K2  spec   ""    ""      2
run spec_fp8m_K2  spec   fp8   marlin  2
echo "[hb] DONE ($(date +%H:%M:%S))"
