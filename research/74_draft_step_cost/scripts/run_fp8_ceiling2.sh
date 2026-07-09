#!/bin/bash
# Follow-up to run_fp8_ceiling.sh. Two jobs:
#
# (1) RE-RUN the 16k fp8blk cell with ITERS=8. The ITERS=4 run tripped the
#     harness's own guardrail (suspect=true, rel-std 15.5% > 15%): per-iter
#     tok/s [619, 623, 452, 652] -- one straggler. base/nospec were clean
#     (<2%). No claim may rest on that cell until it is re-measured.
#
# (2) SAME-BOX MARLIN. The published Marlin numbers (0.68x @ b64) come from
#     cloud-9ezI3Q. To claim "native fp8 (FI-CUTLASS) >> dequant-Marlin" we need
#     Marlin measured HERE, in the same session, against the same bf16 base.
#     `W7_DRAFT_QUANT=fp8` (per-tensor) + `W7_DRAFT_MOE_BACKEND=marlin` is the
#     weight-only W8A16 path (fp8 weights dequant->bf16 MACs).
#
# Guardrail: trust a number only if the logged kernel matches the arm --
#   fp8blk -> FLASHINFER_CUTLASS ; marlin -> MARLIN.
# Run BY PATH (see the pkill hazard note in run_fp8_ceiling.sh).
set -u
PHASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$PHASE/../.." && pwd)"
P52="$REPO/research/52_two_node_e2e"
P57="$REPO/research/57_large_ep_spec_strategy"
PY="$REPO/.venv/bin/python"
ME="$(whoami)"
mkdir -p "$PHASE/logs" "$PHASE/data"

kill_mine(){ for p in 'w7_2node[.]py' 'EngineCor[e]' 'Worker_D[P]'; do pkill -9 -u "$ME" -f "$p" 2>/dev/null; done; sleep 5; }

# run <mode> <arm> <K> <ctx> <batch> <iters>
run(){
  local MODE=$1 ARM=$2 K=$3 CTX=$4 B=$5 IT=$6 MAXLEN=$(( $4 + 4096 ))
  local LOG="$PHASE/logs/fc2_${ARM}_K${K}_ctx${CTX}_b${B}.log"
  kill_mine
  ( source "$PHASE/scripts/env_local.sh"
    unset VLLM_SELF_SPEC_COMPILE_CONSISTENT      # batch-invariant OFF (clean)
    case "$ARM" in
      fp8blk) export W7_DRAFT_QUANT=fp8_per_block ;;                        # native fp8 MACs
      marlin) export W7_DRAFT_QUANT=fp8 W7_DRAFT_MOE_BACKEND=marlin ;;      # W8A16 dequant
    esac
    export W7_KS=$K W7_BATCHES=$B W7_ITERS=$IT W7_WARMUP=1 W7_TAG="p74_fc2_${ARM}_K${K}_ctx${CTX}_b${B}" \
      W7_MASTER_PORT=$((18500 + K + CTX/1000 + RANDOM%20)) W7_CTX_TOKENS="$CTX" W7_MAX_MODEL_LEN="$MAXLEN" \
      W7_MAX_NUM_BATCHED=8192 W7_OUT="$PHASE/data" W7_PROMPT_FILE="$P57/data/prompts_ondist.txt" W7_CHAT=1
    W7_NODE_RANK=0 timeout 3000 "$PY" "$P52/scripts/w7_2node.py" "$MODE" ) > "$LOG" 2>&1
  echo "[fc2] ${ARM} K=${K} ctx${CTX} b${B} iters=${IT}: $(grep -hE 'W7-2N (nospec|K=)' "$LOG" | tail -1)"
  grep -hoE 'Using [A-Za-z_]+ Fp8 MoE backend' "$LOG" | sort -u | sed 's/^/       kernel: /'
  grep -hE 'Traceback|CUDA out of memory|does not support' "$LOG" | tail -2 | sed 's/^/       ERR: /'
  kill_mine
}

run spec fp8blk 2 16384  8 8   # (1) clean re-measure of the suspect cell
run spec marlin 2 16384  8 4   # (2) same-box Marlin, memory-bound
run spec marlin 2  2048 64 4   # (2) same-box Marlin, compute-bound (the 0.68x cell)
echo "[fc2] DONE ($(date +%H:%M:%S))"
