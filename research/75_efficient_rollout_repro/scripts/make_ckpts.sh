#!/bin/bash
# Phase 75 -- build the RTN weight-only drafter checkpoints for Qwen2.5-7B-Instruct.
# CPU, data-free RTN, NO GPU (safe to run while GPUs are reserved).
#
# EfficientRollout §4.1: "apply lightweight RTN quantization to the FFN and QKVO
# projection layers" -> lm_head NOT quantized. `ignore=["lm_head"]` + targets=["Linear"]
# is exactly that set for a dense model.
#
# FIDELITY NOTE: the paper says "the simplest ASYMMETRIC RTN". Our Phase-74 script uses
# symmetric=True. Asymmetric int4 needs zero-points, which changes the kernel path
# (uint4 + zp vs uint4b8). We therefore build BOTH and let S3 report tau for each --
# do not silently substitute one for the other.
#
# Reuses the isolated llmcompressor venv from Phase 74 (built on demand, CPU-only).
# Run BY PATH: `bash scripts/make_ckpts.sh`
set -euo pipefail
PHASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$PHASE/../.." && pwd)"
P74="$REPO/research/74_draft_step_cost"
LC_VENV="${LC_VENV:-$HOME/.cache/eff_lc_venv}"
SRC="${SRC_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
CKPT_ROOT="${CKPT_ROOT:-$HOME/ckpts}"
mkdir -p "$CKPT_ROOT"

if [ ! -x "$LC_VENV/bin/python" ]; then
  echo "[ckpt] creating isolated llmcompressor venv at $LC_VENV (needs uv + PyPI)"
  uv venv --python 3.12 "$LC_VENV"
  "$LC_VENV/bin/python" -m ensurepip --upgrade >/dev/null 2>&1 || true
  uv pip install --python "$LC_VENV/bin/python" llmcompressor transformers torch --torch-backend=cpu
fi
PY="$LC_VENV/bin/python"

build(){  # <script> <out-name> [extra env]
  local script=$1 out=$2
  if [ -d "$CKPT_ROOT/$out" ]; then echo "[ckpt] exists, skip: $CKPT_ROOT/$out"; return; fi
  echo "[ckpt] building $out from $SRC ..."
  SRC_MODEL="$SRC" OUT_DIR="$CKPT_ROOT/$out" "$PY" "$script"
}

# W4A16 int4, group128, SYMMETRIC (-> uint4b8 -> Machete/Marlin fast path)
build "$P74/scripts/make_w4a16_int4_ckpt.py" "Qwen2.5-7B-Instruct-W4A16-INT4-sym"

# W8A16 fp8 weight-only (the paper's W8 comparison row in Tab.4)
build "$P74/scripts/make_w8a16_fp8_ckpt.py" "Qwen2.5-7B-Instruct-W8A16-FP8"

cat <<'EOF'

[ckpt] DONE. Built (symmetric) checkpoints.

STILL TO DO for full paper fidelity -- the ASYMMETRIC W4 variant:
  The paper uses asymmetric RTN. To build it, copy make_w4a16_int4_ckpt.py and set
      symmetric=False           # in QuantizationArgs
  then verify vLLM still routes to an int4 kernel (asym -> zero-points; check the
  engaged kernel in the load log). If asym does NOT route to Machete/Marlin, record
  that and report tau for the symmetric variant, flagging the deviation explicitly.

VERIFY each checkpoint before use:
  - config.json  quantization_config.quant_method == "compressed-tensors"
  - weights are ~1/4 (W4) or ~1/2 (W8) of the bf16 size
  - "lm_head" appears in the ignore list  (paper quantizes FFN + QKVO only)
EOF
