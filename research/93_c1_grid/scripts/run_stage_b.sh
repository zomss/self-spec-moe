#!/bin/bash
# Phase 93 Stage B: e2e serving confirms on real datasets, uncapped
# generation, batch swept — AR + the Stage-A winner arms per arch.
#
# Usage: run_stage_b.sh {dense|mla|moe}
# Env: STAGEB_GPU (dense 6 / mla 7 / moe 6,7), STAGEB_BATCHES
set -uo pipefail
ARCH=${1:?usage: run_stage_b.sh dense|mla|moe}
REPO=/data/smcho/self-spec-moe
PHASE=$REPO/research/93_c1_grid
export HF_HOME=/data/smcho/huggingface
export PATH="$REPO/.venv/bin:$PATH"
cd "$REPO"

BATCHES="${STAGEB_BATCHES:-1,8,32,64}"
CEIL=""

COMMON="VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1 VLLM_SELF_SPEC_CPU_ORCH=1"
SHARED="$COMMON VLLM_SELF_SPEC_SHARED_KV=1 VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE=1 VLLM_SELF_SPEC_SKIP_PREFILL_DRAFT=1 VLLM_SELF_SPEC_DRAFT_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1"
HUM="VLLM_DISABLED_KERNELS=MacheteLinearKernel,CutlassW4A8LinearKernel,AllSparkLinearKernel"
winplain() { echo "$SHARED VLLM_SELF_SPEC_DRAFT_KV_WINDOW=$1 VLLM_SELF_SPEC_DRAFT_KV_SINKS=16"; }

# arm spec: name|draft|K|window|skip|extra-envs
declare -a ARMLIST
# STAGEB_ARMS: optional comma-list of arm names to run (for splitting one
# arch's Stage B across two GPUs). Empty = all.
ARMFILTER="${STAGEB_ARMS:-}"
case "$ARCH" in
  llama)
    MODEL="NousResearch/Meta-Llama-3.1-8B-Instruct"; TP=1
    GPU="${STAGEB_GPU:-7}"; MAXLEN=24576
    SB2=$(python3 -c "import json;print(json.load(open('$PHASE/data/skipsets_llama.json'))['b2'])" 2>/dev/null || echo "3,8")
    ARMLIST=(
      "off|off|0|||"
      "w4a16_k2|/data/smcho/ckpts/Llama31-8B-Instruct-W4A16-INT4-sym|2|||$SHARED"
      "w4a16_k4|/data/smcho/ckpts/Llama31-8B-Instruct-W4A16-INT4-sym|4|||$SHARED"
      "w8int8_k2|/data/smcho/ckpts/Llama31-8B-Instruct-W8A16-INT8-sym|2|||$SHARED"
      "win512_k2|self|2|512||$(winplain 512)"
      "win2048_k4|self|4|2048||$(winplain 2048)"
    )
    ;;
  dense)
    MODEL="Qwen/Qwen3-8B"; TP=1; GPU="${STAGEB_GPU:-6}"; MAXLEN=32768
    ARMLIST=(
      "off|off|0|||"
      "w4a8hum_k2|/data/smcho/ckpts/Qwen3-8B-W4A8-gptq|2|||$SHARED $HUM"
      "w4a8hum_k4|/data/smcho/ckpts/Qwen3-8B-W4A8-gptq|4|||$SHARED $HUM"
      "win2048_k4|self|4|2048||$(winplain 2048)"
      "win512_k2|self|2|512||$(winplain 512)"
      "win128_k2|self|2|128||$(winplain 128)"
    )
    ;;
  mla)
    MODEL="deepseek-ai/DeepSeek-V2-Lite"; TP=1; GPU="${STAGEB_GPU:-7}"; MAXLEN=24576
    # base model loops at T=0 on 8/9 datasets (clip ~1.0 at any ceiling);
    # ceiling 2048 measures the same steady-state cost at sane wall.
    CEIL=2048
    ARMLIST=(
      "off|off|0|||"
      "w8chan_k2|/data/smcho/ckpts/DeepSeek-V2-Lite-W8A16-INT8-chan|2|||$SHARED"
      # added after the cache-key fix: Stage-A originally reported w4a16 at
      # S~0.11 (a 9x slowdown) so it was never promoted to Stage B. Corrected,
      # it wins 13/14 Stage-A cells, so the Stage-B arm set was selected from
      # corrupt data and must include it.
      "w4a16_k2|/data/smcho/ckpts/DeepSeek-V2-Lite-W4A16-INT4-sym|2|||$SHARED VLLM_DISABLED_KERNELS=MarlinLinearKernel"
      "w4a16_k4|/data/smcho/ckpts/DeepSeek-V2-Lite-W4A16-INT4-sym|4|||$SHARED VLLM_DISABLED_KERNELS=MarlinLinearKernel"
      "skipb2_k2|self|2||10,11|$SHARED"
    )
    ;;
  moe)
    MODEL="Qwen/Qwen3-30B-A3B"; TP=2; GPU="${STAGEB_GPU:-6,7}"; MAXLEN=24576
    # filled from the (c)-updated winner map via STAGEB_MOE_ARMS or defaults
    ARMLIST=(
      "off|off|0|||"
      "w4a16_k2|/data/smcho/ckpts/Qwen3-30B-A3B-W4A16-INT4-sym|2|||$SHARED VLLM_DISABLED_KERNELS=MacheteLinearKernel"
      "w4a16_k3|/data/smcho/ckpts/Qwen3-30B-A3B-W4A16-INT4-sym|3|||$SHARED VLLM_DISABLED_KERNELS=MacheteLinearKernel"
      "win2048_k3|self|3|2048||$(winplain 2048)"
      "win8192_k3|self|3|8192||$(winplain 8192)"
    )
    ;;
  q3_32b)
    MODEL="Qwen/Qwen3-32B"; TP=2; GPU="${STAGEB_GPU:-6,7}"; MAXLEN=24576
    SB2=$(python3 -c "import json;print(json.load(open('$PHASE/data/skipsets_q3_32b.json'))['b2'])" 2>/dev/null || echo "7,16")
    # quant e2e arm routed to Machete (W4A16-GPTQ), NOT Humming: the
    # Humming odd-width kernel wedges at TP2 (K4/K6=width 5/7, lazy-cubin
    # livelock, co-tenant-load-sensitive). Humming's faster number stands
    # from Stage A (w4a8hum-K4 1.09-1.51x compiled); this gives a clean
    # uncapped e2e on a non-wedging kernel (conservative vs Humming).
    ARMLIST=(
      "off|off|0|||"
      "w4gptq_k4|/data/smcho/ckpts/Qwen3-32B-W4A16-INT4-gptq|4|||$SHARED"
      "w4gptq_k6|/data/smcho/ckpts/Qwen3-32B-W4A16-INT4-gptq|6|||$SHARED"
      "win512_k4|self|4|512||$(winplain 512)"
      "skipb2_k4|self|4||$SB2|$SHARED"
    )
    ;;
  *) echo "unknown arch"; exit 1;;
esac

gpu_cleanup() {
  for i in ${GPU//,/ }; do
    local u; u=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $i)
    for p in $(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | tr -d ',' | awk -v u="$u" '$2==u{print $1}'); do
      [ "$(ps -o user= -p "$p" 2>/dev/null)" = "smcho" ] || continue
      kill "$p" 2>/dev/null; sleep 2; kill -0 "$p" 2>/dev/null && kill -9 "$p" 2>/dev/null
    done
  done
  sleep 3
}

for spec in "${ARMLIST[@]}"; do
  IFS='|' read -r name draft k window skip extra <<< "$spec"
  if [ -n "$ARMFILTER" ] && ! echo ",$ARMFILTER," | grep -q ",$name,"; then continue; fi
  out="$PHASE/data/stageb_${ARCH}_${name}.json"
  if [ -s "$out" ] && grep -q '"complete": true' "$out"; then echo "[B:$ARCH] skip $name (done)"; continue; fi
  echo "[B:$ARCH] run $name"
  env CUDA_VISIBLE_DEVICES=$GPU G93_MODEL="$MODEL" G93_TP=$TP \
      G93_DRAFT="$draft" G93_K="${k:-0}" G93_WINDOW="${window:-0}" \
      G93_SKIP="$skip" G93_BATCHES="$BATCHES" G93_ITERS=3 \
      G93_CEILING=${CEIL:-16384} G93_MAXLEN=$MAXLEN G93_TAG="stageb_${ARCH}_${name}" \
      G93_OUT="$out" $extra timeout 21600 .venv/bin/python \
      "$PHASE/scripts/run_grid.py" \
      >> "$PHASE/logs/stage_b_${ARCH}.log" 2>&1 \
      || echo "[B:$ARCH] FAIL $name (continuing)"
  gpu_cleanup
done
echo "[B:$ARCH] STAGE-B-${ARCH}-DONE"
