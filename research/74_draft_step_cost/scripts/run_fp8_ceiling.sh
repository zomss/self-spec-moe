#!/bin/bash
# THE fp8 CEILING TEST (the queued "NEXT (when GPUs free)" item).
#
# Claim under test (results_draft_cost.md CORRECTION block): every fp8 number in
# Phase 74 used per-tensor `quantization=fp8`, which routes to the taxed TRITON
# W8A8 (0.65x) or dequant-MARLIN (bf16 MACs, parity). The NATIVE fp8 tensor-core
# path -- `quantization=fp8_per_block` -> FlashInfer-CUTLASS, auto-picked on
# Hopper for block-fp8 with ep_size>1 (oracle/fp8.py:160-170) -- was NEVER
# measured. On-the-fly from bf16: no checkpoint, no DeepGEMM.
#
# WHY the reference arms are re-run here: the published refs (nospec 599,
# bf16 base 592 @16k b8) were measured on a DIFFERENT box (cloud-9ezI3Q).
# Cross-box tok/s is not comparable, so nospec + bf16 base are re-measured in
# this same session on this box. Only same-session ratios are reported.
#
# Guardrail: an fp8 number is TRUSTED ONLY IF the log shows
#   "Using flashinfer_cutlass Fp8 MoE backend"
# Any triton/marlin fallback means we measured the taxed path again, not the ceiling.
set -u
PHASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$PHASE/../.." && pwd)"
P52="$REPO/research/52_two_node_e2e"
P57="$REPO/research/57_large_ep_spec_strategy"
PY="$REPO/.venv/bin/python"
ME="$(whoami)"
mkdir -p "$PHASE/logs" "$PHASE/data"

# Only ever our OWN processes (shared node).
# HAZARD: `pkill -f` matches against full command lines, so it will SIGKILL any
# shell whose own argv contains these patterns. Run this file BY PATH
# (`bash scripts/run_fp8_ceiling.sh`); never paste its body into a heredoc or a
# `bash -c` string, or the invoking shell matches and kills itself.
kill_mine(){ for p in 'w7_2node[.]py' 'EngineCor[e]' 'Worker_D[P]'; do pkill -9 -u "$ME" -f "$p" 2>/dev/null; done; sleep 5; }

# run <mode> <arm> <K> <ctx> <batch>
run(){
  local MODE=$1 ARM=$2 K=$3 CTX=$4 B=$5 MAXLEN=$(( $4 + 4096 ))
  local LOG="$PHASE/logs/fc_${ARM}_K${K}_ctx${CTX}_b${B}.log"
  kill_mine
  ( source "$PHASE/scripts/env_local.sh"
    unset VLLM_SELF_SPEC_COMPILE_CONSISTENT      # batch-invariant OFF (clean)
    # NOTE: no VLLM_SELF_SPEC_PROFILE -> no per-region cuda.synchronize
    case "$ARM" in
      fp8blk)  export W7_DRAFT_QUANT=fp8_per_block ;;   # native; NO moe_backend force
      fp8chan) export W7_DRAFT_QUANT=fp8_per_channel ;;
      window)  export VLLM_SELF_SPEC_DRAFT_KV_WINDOW=512 VLLM_SELF_SPEC_DRAFT_KV_SINKS=16 ;;
    esac
    export W7_KS=$K W7_BATCHES=$B W7_ITERS=4 W7_WARMUP=1 W7_TAG="p74_fc_${ARM}_K${K}_ctx${CTX}_b${B}" \
      W7_MASTER_PORT=$((18300 + K + CTX/1000 + RANDOM%20)) W7_CTX_TOKENS="$CTX" W7_MAX_MODEL_LEN="$MAXLEN" \
      W7_MAX_NUM_BATCHED=8192 W7_OUT="$PHASE/data" W7_PROMPT_FILE="$P57/data/prompts_ondist.txt" W7_CHAT=1
    W7_NODE_RANK=0 timeout 2400 "$PY" "$P52/scripts/w7_2node.py" "$MODE" ) > "$LOG" 2>&1
  echo "[fc] ${ARM} K=${K} ctx${CTX} b${B}: $(grep -hE 'W7-2N (nospec|K=)' "$LOG" | tail -1)"
  # Which fp8 MoE kernel actually engaged? (blank for bf16 arms, as expected).
  # The oracle logs the enum name UPPERCASE (e.g. FLASHINFER_CUTLASS).
  grep -hoE 'Using [A-Za-z_]+ Fp8 MoE backend' "$LOG" | sort -u | sed 's/^/       kernel: /'
  grep -hE 'Traceback|CUDA out of memory|does not support' "$LOG" | tail -2 | sed 's/^/       ERR: /'
  kill_mine
}

# Cell 1 -- memory-bound (the queued item): 16k, b8, K=2.
# Prediction from the "unifying principle": weight-read is NOT the binding axis
# here, so even native fp8 should land at ~parity (a draft-only lever on the
# wrong axis). A win here would falsify that model.
if [ "${CELL:-both}" != compute ]; then
  run nospec nospec 2 16384 8      # same-box denominator
  run spec   base   2 16384 8      # same-box bf16 draft reference
  run spec   fp8blk 2 16384 8      # THE new datapoint: native fp8 tensor-core MoE
fi

# Cell 2 -- compute-bound: 2k, b64, K=2. This is the cell that DISCRIMINATES
# native fp8 from Marlin. "WHY fp8 <= bf16" argued Marlin cannot accelerate
# compute (it dequants to bf16 MACs) and so must lose when compute-bound:
# measured bf16 0.78x vs Marlin 0.68x. FI-CUTLASS does real fp8xfp8 MACs, so if
# that mechanism is right, fp8blk should now beat Marlin and approach/exceed
# bf16's 0.78x. (Self-spec still expected < 1.0x here -- compute-saturated.)
if [ "${CELL:-both}" != memory ]; then
  run nospec nospec 2 2048 64
  run spec   base   2 2048 64
  run spec   fp8blk 2 2048 64
fi
echo "[fc] DONE ($(date +%H:%M:%S))"
