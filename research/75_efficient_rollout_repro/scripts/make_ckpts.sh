#!/bin/bash
# Phase 75 -- build the RTN weight-only drafter checkpoints for Qwen2.5-7B-Instruct.
# CPU, data-free RTN, NO GPU (safe to run while GPUs are reserved). ~15-25 min.
#
# EfficientRollout 4.1 quantizes the FFN + QKVO projections and leaves lm_head alone;
# targets=["Linear"] + ignore=["lm_head"] is exactly that set for a dense model.
#
# Builds THREE checkpoints:
#   W4A16-INT4-sym   -> uint4b8   (Machete/Marlin fast path)   <- E1/E2 primary
#   W8A16-INT8-sym   -> uint8b128 (SAME kernel family as W4)    <- clean bit-width A/B
#   W4A16-INT4-asym  -> uint4 + zero-points                     <- the paper's "asymmetric RTN"
#
# Why int8 (not fp8) for the W8 arm: it keeps W4 and W8 on ONE kernel family, so the
# W4-vs-W8 comparison isolates bit-width. An fp8 W8A16 arm would route to FP8-Marlin
# behind a separate gate and confound the comparison.
#
# Run BY PATH: `bash scripts/make_ckpts.sh`
set -euo pipefail
PHASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LC_VENV="${LC_VENV:-$HOME/.cache/eff_lc_venv}"
SRC="${SRC_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
CKPT_ROOT="${CKPT_ROOT:-/data/smcho/ckpts}"
MK="$PHASE/scripts/make_wxa16_ckpt.py"
mkdir -p "$CKPT_ROOT"

if [ ! -x "$LC_VENV/bin/python" ]; then
  echo "[ckpt] creating isolated llmcompressor venv at $LC_VENV (needs uv + PyPI)"
  uv venv --python 3.12 "$LC_VENV"
  uv pip install --python "$LC_VENV/bin/python" llmcompressor transformers --torch-backend=cpu
fi
PY="$LC_VENV/bin/python"
NAME="$(basename "$SRC")"

build(){  # <bits> <sym:1|0>
  local bits=$1 sym=$2
  local tag="sym"; [ "$sym" = 0 ] && tag="asym"
  local out="$CKPT_ROOT/${NAME}-W${bits}A16-INT${bits}-${tag}"
  if [ -d "$out" ]; then echo "[ckpt] exists, skip: $out"; return; fi
  SRC_MODEL="$SRC" BITS="$bits" SYM="$sym" OUT_DIR="$out" "$PY" "$MK"
}

build 4 1     # primary
build 8 1     # bit-width A/B on the same kernel
build 4 0     # paper fidelity: asymmetric RTN

echo
echo "[ckpt] built under $CKPT_ROOT:"
ls -d "$CKPT_ROOT/${NAME}"-W*A16-* 2>/dev/null || true
cat <<'EOF'

VERIFY before trusting any number:
  - config.json: quantization_config.quant_method == "compressed-tensors"
  - "lm_head" present in the ignore list   (paper quantizes FFN + QKVO only)
  - safetensors size ~= 1/4 (W4) or ~1/2 (W8) of the bf16 model

Then check the kernel each will actually use (the chooser logs NOTHING, so ask it):
  python scripts/which_kernel.py --bits 4                 # expect MacheteLinearKernel
  python scripts/which_kernel.py --bits 4 --zero-points   # asym -> may differ!
  VLLM_DISABLED_KERNELS=MacheteLinearKernel,CutlassW4A8LinearKernel,AllSparkLinearKernel \
    python scripts/which_kernel.py --bits 4               # expect MarlinLinearKernel

If the ASYM variant reports KERNEL=NONE, say so explicitly in results_repro.md and fall
back to sym -- do NOT silently substitute one for the other.
EOF
