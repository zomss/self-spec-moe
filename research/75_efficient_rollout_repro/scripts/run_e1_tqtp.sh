#!/bin/bash
# Phase 75 -- E1 (KEY #1): the MEASURED draft/target cost ratio Tq/Tp.
#
# EfficientRollout Tab.4 reports Tq/Tp = 0.360 (W4) / 0.573 (W8), but those are
# "predicted by our roofline model under memory-bound, ZERO-OVERHEAD assumptions".
# This measures them.
#
# Method: `vllm bench latency` at two output lengths; the SLOPE removes prefill, so
#   per-decode-step ms = (T(O2) - T(O1)) / (O2 - O1)
# Arms: bf16 / W8A16-int8 / W4A16-int4, x kernel {Machete (auto), Marlin (paper's)}.
# Both quant widths ride the SAME mixed-precision kernel family (Machete supports
# uint4b8 and uint8b128), so W4-vs-W8 is a clean bit-width comparison.
#
# NOTE (documented, deliberate): this times a standalone CUDA-graphed forward. The real
# self-spec draft chain runs PIECEWISE/eager, so this ratio is a LOWER BOUND (optimistic).
# E3 measures the truth. E1-pass + E3-fail => the lever is real and our harness eats it.
#
# Run BY PATH: `bash scripts/run_e1_tqtp.sh`
set -u
PHASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$PHASE/../.." && pwd)"
PY="$REPO/.venv/bin/python"
CKPT_ROOT="${CKPT_ROOT:-$HOME/ckpts}"
BASE="${E75_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
W4="${E75_W4:-$CKPT_ROOT/Qwen2.5-7B-Instruct-W4A16-INT4-sym}"
W8="${E75_W8:-$CKPT_ROOT/Qwen2.5-7B-Instruct-W8A16-INT8-sym}"
CTX="${E75_CTX:-2048}"
O1=1; O2=33                 # slope over 32 decode steps
ITERS="${E75_ITERS:-10}"; WARM=3
mkdir -p "$PHASE/logs" "$PHASE/data"
ME="$(whoami)"
kill_mine(){ pkill -9 -u "$ME" -f 'vllm[.]entrypoints' 2>/dev/null; sleep 3; }

arm(){   # <name> <model> <kernel:auto|marlin> <bits|0>
  local name=$1 model=$2 kern=$3 bits=$4
  if [ ! -e "$model" ] && [[ "$model" == /* ]]; then
    echo "[e1] SKIP $name -- checkpoint missing: $model (run make_ckpts.sh)"; return
  fi
  ( source "$PHASE/scripts/env_e75.sh"
    if [ "$kern" = marlin ]; then e75_force_marlin; else e75_auto_kernel; fi

    # Which kernel will actually run? The chooser logs nothing, so ask it directly.
    if [ "$bits" != 0 ]; then
      K=$("$PY" "$PHASE/scripts/which_kernel.py" --bits "$bits" 2>/dev/null | grep '^KERNEL=' | cut -d= -f2)
      echo "[e1] $name kernel=$K"
      case "$kern:$K" in
        auto:MacheteLinearKernel|marlin:MarlinLinearKernel) ;;
        *) echo "[e1] ABORT $name -- wanted $kern, chooser says $K"; exit 0 ;;
      esac
    else
      echo "[e1] $name kernel=bf16(no MP kernel)"
    fi

    for O in $O1 $O2; do
      J="$PHASE/data/e1_${name}_o${O}.json"
      L="$PHASE/logs/e1_${name}_o${O}.log"
      "$PY" -m vllm.entrypoints.cli.main bench latency \
        --model "$model" --input-len "$CTX" --output-len "$O" --batch-size 1 \
        --max-model-len $((CTX + 512)) --gpu-memory-utilization 0.85 \
        --num-iters-warmup "$WARM" --num-iters "$ITERS" \
        --output-json "$J" > "$L" 2>&1 \
        || { echo "[e1] FAIL $name o=$O -> $L"; grep -hE 'Error|Traceback|out of memory' "$L" | tail -2; }
    done
  )
  kill_mine
}

echo "[e1] START ($(date +%H:%M:%S))  ctx=$CTX  slope over $((O2-O1)) decode steps"
arm bf16       "$BASE" auto   0
arm w8_machete "$W8"   auto   8
arm w4_machete "$W4"   auto   4
arm w8_marlin  "$W8"   marlin 8
arm w4_marlin  "$W4"   marlin 4
echo "[e1] runs done; analyzing"
"$PY" "$PHASE/scripts/analyze_e1.py" --model "$BASE" --ctx "$CTX" --o1 $O1 --o2 $O2 \
  | tee "$PHASE/logs/e1_summary.txt"
echo "[e1] DONE ($(date +%H:%M:%S))  -> logs/e1_summary.txt"
