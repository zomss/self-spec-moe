#!/bin/bash
# Third leg: the 32k half of the queued fp8-ceiling item (K*=3 at 32k per the
# K-tuning result). Same-box nospec + bf16 base + native fp8 (FI-CUTLASS).
#
# Why 32k matters on its own: it is where window WINS 1.34x and where global
# KV-quant degrades MONOTONICALLY (0.95x@16k -> 0.89x@32k). If native fp8 were
# ever going to help, a longer context makes the draft more KV-bound and less
# fixed-overhead-bound -- the same escape hatch that KV-quant failed. Expect
# parity again (weight/MAC is still not the binding axis).
#
# ITERS=6: the 16k fp8 cell was bimodal (~0.6s stall on ~40% of long decodes),
# so give the median enough samples to be stable.
# Guardrail: trust fp8 numbers only if the log shows FLASHINFER_CUTLASS.
# Run BY PATH (pkill -f self-match hazard; see run_fp8_ceiling.sh).
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
  local LOG="$PHASE/logs/fc3_${ARM}_K${K}_ctx${CTX}_b${B}.log"
  kill_mine
  ( source "$PHASE/scripts/env_local.sh"
    unset VLLM_SELF_SPEC_COMPILE_CONSISTENT
    case "$ARM" in
      fp8blk) export W7_DRAFT_QUANT=fp8_per_block ;;
    esac
    export W7_KS=$K W7_BATCHES=$B W7_ITERS=$IT W7_WARMUP=1 W7_TAG="p74_fc3_${ARM}_K${K}_ctx${CTX}_b${B}" \
      W7_MASTER_PORT=$((18700 + K + CTX/1000 + RANDOM%20)) W7_CTX_TOKENS="$CTX" W7_MAX_MODEL_LEN="$MAXLEN" \
      W7_MAX_NUM_BATCHED=8192 W7_OUT="$PHASE/data" W7_PROMPT_FILE="$P57/data/prompts_ondist.txt" W7_CHAT=1
    W7_NODE_RANK=0 timeout 3600 "$PY" "$P52/scripts/w7_2node.py" "$MODE" ) > "$LOG" 2>&1
  echo "[fc3] ${ARM} K=${K} ctx${CTX} b${B} iters=${IT}: $(grep -hE 'W7-2N (nospec|K=)' "$LOG" | tail -1)"
  grep -hoE 'Using [A-Za-z_]+ Fp8 MoE backend' "$LOG" | sort -u | sed 's/^/       kernel: /'
  grep -hE 'Traceback|CUDA out of memory|does not support' "$LOG" | tail -2 | sed 's/^/       ERR: /'
  kill_mine
}

run nospec nospec 3 32768 8 6
run spec   base   3 32768 8 6
run spec   fp8blk 3 32768 8 6
echo "[fc3] DONE ($(date +%H:%M:%S))"
