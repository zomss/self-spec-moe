#!/usr/bin/env bash
# W1b: source-B discriminator -- is the bimodal ~126/~142 AR mode CPU
# placement? AR-only (arm=off), notune (source A controlled), BOTH GPU
# lanes, pin vs nopin interleaved, 3 boots each = 12 boots.
#
# Pre-registered (before any boot):
#   P-W1b1  pinned boots are unimodal (<5% between-boot swing) on both
#           lanes -> source B = CPU core placement/contention; the fix is
#           to pin every future run; W1 gate satisfiable.
#   P-W1b2  the mode appears on GPU1's lane but not GPU0's (as in the
#           first matrix, 2/6 vs 0/6) -> lane-specific state, not chance.
#   P-W1b3  pinned boots still bimodal -> cause is inside the engine boot
#           (allocator/JIT state); llama AR anchors get an N-boot
#           median protocol and that is DISCLOSED.
#
# Pinning: disjoint quiet core sets per lane (engine TP1 fits easily):
#   GPU0 lane -> cores 0-15, GPU1 lane -> cores 16-31.
set -uo pipefail
cd /data/smcho/self-spec-moe

PHASE=research/96_selector_foundations
LOG="$PHASE/logs"; mkdir -p "$LOG" "$PHASE/data/w1"

export PATH="/data/smcho/self-spec-moe/.venv/bin:$PATH"
export HF_HOME=/data/smcho/huggingface
export VLLM_CACHE_ROOT=/data/smcho/.cache/vllm
export TRITON_CACHE_DIR=/data/smcho/.cache/triton
export TMPDIR=/data/smcho/tmp

cleanup() {
  local gpu="$1"
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$gpu" 2>/dev/null); do
    kill -9 "$p" 2>/dev/null
  done
  local waited=0
  while [ $waited -lt 180 ]; do
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" 2>/dev/null)
    [ "${used:-0}" -le 2000 ] && break
    sleep 10; waited=$((waited + 10))
  done
  sleep 5
}

lane() {      # gpu  cores
  local gpu="$1" cores="$2"
  for boot in 0 1 2; do
    for pin in pin nopin; do
      local tag="off_notune_g${gpu}_${pin}_b${boot}"
      local out="$PHASE/data/w1/w1b_llama_${tag}.json"
      if [ -f "$out" ]; then echo "[W1b] skip $tag (exists)"; continue; fi
      echo "[W1b] === gpu=$gpu $tag ==="
      cleanup "$gpu"
      local wrap=""
      [ "$pin" = pin ] && wrap="taskset -c $cores"
      env CUDA_VISIBLE_DEVICES="$gpu" \
          W1_ARM=off W1_TUNE=0 W1_BOOT="$boot" W1_OUT="$out" W1_PIN="$([ "$pin" = pin ] && echo "$cores")" \
          timeout 1800 $wrap .venv/bin/python \
            "$PHASE/scripts/run_w1_boot.py" \
            >> "$LOG/w1b_g${gpu}.log" 2>&1
      [ $? -ne 0 ] && echo "[W1b] FAILED $tag (see $LOG/w1b_g${gpu}.log)"
    done
  done
  cleanup "$gpu"
  echo "[W1b] lane done gpu=$gpu"
}

lane 0 0-15 &
L0=$!
lane 1 16-31 &
L1=$!
wait $L0 $L1
echo "[W1b] matrix complete"
ls -la "$PHASE/data/w1/" | grep w1b
