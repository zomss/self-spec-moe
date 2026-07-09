#!/bin/bash
# Build the dense W4A16-int4 checkpoint reusing the EXISTING isolated lc_venv (no
# reinstall). CPU / data-free RTN -> no reserved-GPU contention.
set -u
PHASE=/data/smcho/self-spec-moe/research/74_draft_step_cost
LCVENV=${LC_VENV:-$HOME/.cache/eff_lc_venv}   # portable; must match make_ckpt.sh (run it first)
OUT_DIR="$HOME/ckpts/Qwen3-8B-W4A16-INT4"
LOG="$PHASE/logs/make_int4_ckpt.log"; : > "$LOG"
[ -x "$LCVENV/bin/python" ] || { echo "[ckpt] lc_venv missing; run make_ckpt.sh first" | tee -a "$LOG"; exit 1; }
echo "[ckpt] running data-free RTN W4A16-int4 on CPU ..." | tee -a "$LOG"
CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 \
  SRC_MODEL="Qwen/Qwen3-8B" OUT_DIR="$OUT_DIR" \
  "$LCVENV/bin/python" "$PHASE/scripts/make_w4a16_int4_ckpt.py" >>"$LOG" 2>&1
RC=$?
echo "[ckpt] exit=$RC  out=$OUT_DIR" | tee -a "$LOG"
[ -d "$OUT_DIR" ] && ls -la "$OUT_DIR" 2>/dev/null | awk '{print $5,$9}' | tee -a "$LOG"
echo "[ckpt] DONE ($(date +%H:%M:%S))" | tee -a "$LOG"
