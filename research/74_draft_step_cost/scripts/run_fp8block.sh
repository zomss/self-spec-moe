#!/bin/bash
# REACH THE WEIGHT-QUANT BAR: native fp8 tensor-core MoE via on-the-fly 128-block
# fp8 (quantization=fp8_per_block) -> FlashInfer-CUTLASS (auto-picked for Hopper
# block-fp8 EP>1). NO checkpoint, NO DeepGEMM (flashinfer 0.6.12 already installed).
# vs the taxed per-tensor paths measured before (fp8+triton 383, fp8+marlin ~592).
# Expectation: native fp8 = clean weight-read halving, no dequant tax, no per-tensor
# a2a scale-gather -> could turn parity into a modest win at low batch.
# Refs (bf16 base, no window): 16k K=2 = 592, 32k K=3 = 586; no-spec 16k=599 32k=436.
# DO NOT run while GPUs 4-7 are reserved by others.
set -u
PHASE=/data/smcho/self-spec-moe/research/74_draft_step_cost
P52=/data/smcho/self-spec-moe/research/52_two_node_e2e
P57=/data/smcho/self-spec-moe/research/57_large_ep_spec_strategy
PY=/data/smcho/self-spec-moe/.venv/bin/python
ME="$(whoami)"
kill_mine(){ for p in 'w7_2node[.]py' 'EngineCor[e]' 'Worker_D[P]'; do pkill -9 -u "$ME" -f "$p" 2>/dev/null; done; sleep 5; }
run(){  # quant K ctx
  local Q=$1 K=$2 CTX=$3 MAXLEN=$(( $3 + 4096 ))
  local LOG="$PHASE/logs/fp8blk_${Q}_K${K}_ctx${CTX}_b8.log"
  kill_mine
  ( source "$PHASE/scripts/env_cloud4.sh"
    unset VLLM_SELF_SPEC_COMPILE_CONSISTENT
    export W7_DRAFT_QUANT="$Q"          # fp8_per_block (native) — NO moe_backend force
    export W7_KS=$K W7_BATCHES=8 W7_ITERS=4 W7_WARMUP=1 W7_TAG="p74_fp8blk_${Q}_K${K}_ctx${CTX}_b8" \
      W7_MASTER_PORT=$((18100 + K + CTX/1000 + RANDOM%20)) W7_CTX_TOKENS="$CTX" W7_MAX_MODEL_LEN="$MAXLEN" \
      W7_MAX_NUM_BATCHED=8192 W7_OUT="$PHASE/data" W7_PROMPT_FILE="$P57/data/prompts_ondist.txt" W7_CHAT=1
    W7_NODE_RANK=0 timeout 1200 "$PY" "$P52/scripts/w7_2node.py" spec ) > "$LOG" 2>&1
  echo "[fp8blk] ${Q} K${K} ctx${CTX}: $(grep -hE 'W7-2N.*batch=' "$LOG" | tail -1)"
  # CONFIRM the native backend engaged (want FLASHINFER_CUTLASS, NOT TRITON/MARLIN)
  grep -hiE 'Fp8 MoE backend|FLASHINFER_CUTLASS|flashinfer.*cutlass|block' "$LOG" | tail -3
  kill_mine
}
run fp8_per_block 2 16384
run fp8_per_block 3 32768
# optional 2nd native option:
# run fp8_per_channel 2 16384
echo "[fp8blk] DONE ($(date +%H:%M:%S))"
