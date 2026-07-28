#!/bin/bash
# Phase 93 Step 1a: build the MoE + MLA drafter checkpoint ladder.
# CPU-only (data-free RTN / FP8-dynamic), sequential to bound RAM.
# Completeness-gated like phase-92's make_fresh_drafter (tmp+rename).
set -euo pipefail
PHASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LC_VENV="${LC_VENV:-$HOME/.cache/eff_lc_venv}"
CKPT_ROOT="${CKPT_ROOT:-$HOME/ckpts}"
export HF_HOME=/data/smcho/huggingface
PY="$LC_VENV/bin/python"

build() {  # <script> <src> <out-name> [extra env as K=V ...]
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

MOE="Qwen/Qwen3-30B-A3B"
MLA="deepseek-ai/DeepSeek-V2-Lite"

build make_wxa16_93.py "$MOE" "Qwen3-30B-A3B-W4A16-INT4-sym" BITS=4 SYM=1
# W8 MoE must be CHANNEL-wise (Marlin-MoE int8 asserts group_size==-1)
build make_wxa16_93.py "$MOE" "Qwen3-30B-A3B-W8A16-INT8-chan" BITS=8 SYM=1 GROUP=-1
build make_fp8_dynamic.py "$MOE" "Qwen3-30B-A3B-FP8-dynamic"
# V2-Lite layer-0 (the only dense MLP; first_k_dense_replace=1) has
# down_proj columns=10944, not divisible by group 128 -> stays bf16
# (disclosed). MoE expert projections (1408) are divisible.
MLA_IGNORE='lm_head,re:.*mlp.gate$,re:.*shared_expert_gate$,re:.*layers.0.mlp.down_proj$'
build make_wxa16_93.py "$MLA" "DeepSeek-V2-Lite-W4A16-INT4-sym" BITS=4 SYM=1 IGNORE="$MLA_IGNORE"
# V2-Lite experts hit the same Marlin-MoE int8 channel-wise assert
build make_wxa16_93.py "$MLA" "DeepSeek-V2-Lite-W8A16-INT8-chan" BITS=8 SYM=1 GROUP=-1 IGNORE="$MLA_IGNORE"
build make_fp8_dynamic.py "$MLA" "DeepSeek-V2-Lite-FP8-dynamic" IGNORE="$MLA_IGNORE"

echo "[93-ckpt] ALL DONE"
ls -d "$CKPT_ROOT"/Qwen3-30B-A3B-* "$CKPT_ROOT"/DeepSeek-V2-Lite-* 2>/dev/null
