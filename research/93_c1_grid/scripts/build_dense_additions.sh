#!/bin/bash
# Phase 93 Step 1a addendum: dense-8B ladder gaps (W8A16-INT8, FP8-dynamic).
set -euo pipefail
PHASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LC_VENV="${LC_VENV:-$HOME/.cache/eff_lc_venv}"
CKPT_ROOT="${CKPT_ROOT:-/data/smcho/ckpts}"
export HF_HOME=/data/smcho/huggingface
PY="$LC_VENV/bin/python"

build() {
  local script=$1 src=$2 name=$3; shift 3
  local out="$CKPT_ROOT/$name"
  if [ -d "$out" ] && grep -q quantization_config "$out/config.json" 2>/dev/null; then
    echo "[93-ckpt] exists (complete): $name"; return 0
  fi
  rm -rf "$out" "$out.tmp"
  echo "[93-ckpt] building $name ..."
  env "$@" SRC_MODEL="$src" OUT_DIR="$out.tmp" "$PY" "$PHASE/scripts/$script"
  grep -q quantization_config "$out.tmp/config.json"
  mv "$out.tmp" "$out"
  echo "[93-ckpt] DONE $name"
}

build make_wxa16_93.py "Qwen/Qwen3-8B" "Qwen3-8B-W8A16-INT8-sym" BITS=8 SYM=1
build make_fp8_dynamic.py "Qwen/Qwen3-8B" "Qwen3-8B-FP8-dynamic"
echo "[93-ckpt] dense additions DONE"
