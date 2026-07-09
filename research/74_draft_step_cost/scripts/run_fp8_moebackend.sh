#!/bin/bash
# Force the fp8 DRAFT MoE backend (auto picked slow TRITON w/o DeepGEMM).
# fp8 draft @16k b8, COMPILE_CONSISTENT off, try flashinfer_trtllm / cutlass.
# Compare to TRITON fp8 (291 tok/s) and base bf16 (370).
set -u
PHASE=/data/smcho/self-spec-moe/research/74_draft_step_cost
P52=/data/smcho/self-spec-moe/research/52_two_node_e2e
P57=/data/smcho/self-spec-moe/research/57_large_ep_spec_strategy
PY=/data/smcho/self-spec-moe/.venv/bin/python
ME="$(whoami)"
kill_mine(){ for p in 'w7_2node[.]py' 'EngineCor[e]' 'Worker_D[P]'; do pkill -9 -u "$ME" -f "$p" 2>/dev/null; done; sleep 5; }
for MB in ${MBS:-flashinfer_trtllm flashinfer_cutlass}; do
  PROFDIR="$PHASE/data/prof_fp8mb_${MB}_16k_b8"; mkdir -p "$PROFDIR"; rm -f "$PROFDIR"/*.json 2>/dev/null
  LOG="$PHASE/logs/fp8mb_${MB}_16k_b8.log"
  kill_mine
  ( source "$PHASE/scripts/env_cloud4.sh"
    unset VLLM_SELF_SPEC_COMPILE_CONSISTENT
    export W7_DRAFT_QUANT=fp8 W7_DRAFT_MOE_BACKEND="$MB"
    export W7_KS=4 W7_BATCHES=8 W7_ITERS=2 W7_WARMUP=1 W7_TAG="p74_fp8mb_${MB}_16k_b8" \
      W7_MASTER_PORT=$((16800 + RANDOM%50)) W7_CTX_TOKENS=16384 W7_MAX_MODEL_LEN=20480 \
      W7_MAX_NUM_BATCHED=8192 W7_OUT="$PHASE/data" W7_PROMPT_FILE="$P57/data/prompts_ondist.txt" W7_CHAT=1 \
      VLLM_SELF_SPEC_PROFILE=1 VLLM_SELF_SPEC_PROFILE_OUT="$PROFDIR" VLLM_SELF_SPEC_PROFILE_WARMUP=20
    W7_NODE_RANK=0 timeout 1200 "$PY" "$P52/scripts/w7_2node.py" spec ) > "$LOG" 2>&1
  echo "[fp8mb] $MB 16k b8:"
  grep -hE "W7-2N.*batch=|Fp8 MoE backend|MoEPrepareAndFinalize|Traceback|Error|not support|assert" "$LOG" | tail -4
  kill_mine
done
echo "[fp8mb] DONE ($(date +%H:%M:%S))"
