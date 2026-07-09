#!/bin/bash
# Produce the dense W8A16-fp8 checkpoint (Item 1) in an ISOLATED venv (does NOT touch
# the main vLLM env) on CPU (CUDA_VISIBLE_DEVICES="" -> no reserved-GPU contention).
set -u
PHASE=/data/smcho/self-spec-moe/research/74_draft_step_cost
LCVENV=/tmp/claude-1001/-data-smcho-self-spec-moe/ce0c3688-a79b-4d56-8190-24141d0e2a99/scratchpad/lc_venv
OUT_DIR="$HOME/ckpts/Qwen3-8B-W8A16-FP8"
LOG="$PHASE/logs/make_ckpt.log"
: > "$LOG"
echo "[ckpt] disk free:" | tee -a "$LOG"; df -h "$HOME" /tmp 2>/dev/null | tee -a "$LOG"
echo "[ckpt] creating isolated venv at $LCVENV ..." | tee -a "$LOG"
uv venv --python 3.12 "$LCVENV" >>"$LOG" 2>&1 || { echo "[ckpt] venv FAILED" | tee -a "$LOG"; exit 1; }
echo "[ckpt] installing llmcompressor (PyPI) ..." | tee -a "$LOG"
uv pip install --python "$LCVENV/bin/python" llmcompressor >>"$LOG" 2>&1 || { echo "[ckpt] pip install FAILED (network?)" | tee -a "$LOG"; exit 2; }
echo "[ckpt] running data-free RTN W8A16-fp8 on CPU ..." | tee -a "$LOG"
CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 \
  SRC_MODEL="Qwen/Qwen3-8B" OUT_DIR="$OUT_DIR" \
  "$LCVENV/bin/python" "$PHASE/scripts/make_w8a16_fp8_ckpt.py" >>"$LOG" 2>&1
RC=$?
echo "[ckpt] exit=$RC  out=$OUT_DIR" | tee -a "$LOG"
[ -d "$OUT_DIR" ] && ls -la "$OUT_DIR" | tee -a "$LOG"
echo "[ckpt] DONE ($(date +%H:%M:%S))" | tee -a "$LOG"
