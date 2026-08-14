#!/bin/bash
# Phase 93 Step 3 / Stage A: compiled cell grids per arch x arm.
# One engine boot per (arm, K), decode cost via T(1+N)-T(1) on identical
# prompts over batch x ctx cells (82/88 compile protocol). Resumable:
# skips (arm, K) whose rows already exist in the per-arm CSV.
#
# Usage: run_stage_a.sh {dense|mla|moe}
# GPUs: dense/mla -> one of 6/7 via STAGEA_GPU; moe -> 6,7 (TP2).
set -uo pipefail

ARCH=${1:?usage: run_stage_a.sh dense|mla|moe}
REPO=/data/smcho/self-spec-moe
PHASE=$REPO/research/93_c1_grid
P82DATA=$REPO/research/82_runtime_switching/data
COMPILE=$PHASE/scripts/compile_cells_93.py
export HF_HOME=/data/smcho/huggingface
export PATH="$REPO/.venv/bin:$PATH"
cd "$REPO"

BATCHES="1,4,8,16,32,64,128"
CTXS="2000,8000,14000"
KS="k2 k4 k6"

# Production-stack env bundles (the "fixed stack" every winning record
# number used -- T5/T6/T8 arms ran WITH these; without them drafts pay
# naive-path costs e.g. their own prefill. Discovered the hard way:
# the first Stage-A pass measured OFF-everywhere).
COMMON="VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1 VLLM_SELF_SPEC_CPU_ORCH=1"
# shared-KV full-context stack (quant + skip arms): target-KV binding,
# no draft prefill, piecewise chain (FULLCG needs a window -> not here;
# the plain-chain cost IS the honest full-ctx realization)
SHARED="$COMMON VLLM_SELF_SPEC_SHARED_KV=1 VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE=1 VLLM_SELF_SPEC_SKIP_PREFILL_DRAFT=1 VLLM_SELF_SPEC_DRAFT_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1"
# windowed scratchpad stack (GQA window arms): + FULLCG + wholechain
winstack() { echo "$SHARED VLLM_SELF_SPEC_DRAFT_KV_WINDOW=$1 VLLM_SELF_SPEC_DRAFT_KV_SINKS=16 VLLM_SELF_SPEC_DRAFT_FULLCG=1"; }
# MLA window arms: scratchpad/FULLCG is GQA-only -> plain chain + window
winplain() { echo "$SHARED VLLM_SELF_SPEC_DRAFT_KV_WINDOW=$1 VLLM_SELF_SPEC_DRAFT_KV_SINKS=16"; }
# kvq: draft OWNS its KV (fp8) -> no shared-KV, draft pays prefill
KVQSTACK="$COMMON VLLM_SELF_SPEC_DRAFT_KV_DTYPE=fp8 VLLM_SELF_SPEC_DRAFT_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1"

declare -A ARMS
case "$ARCH" in
  dense)
    MODEL="Qwen/Qwen3-8B"; TP=1; KVLIM=260000
    GPU="${STAGEA_GPU:-6}"
    ARMS[w4a16]="/data/smcho/ckpts/Qwen3-8B-W4A16-INT4|$SHARED"
    ARMS[w4a8cut]="/data/smcho/ckpts/Qwen3-8B-W4A8-gptq|$SHARED"
    ARMS[w4a8hum]="/data/smcho/ckpts/Qwen3-8B-W4A8-gptq|$SHARED VLLM_DISABLED_KERNELS=MacheteLinearKernel,CutlassW4A8LinearKernel,AllSparkLinearKernel"
    ARMS[w8int8]="/data/smcho/ckpts/Qwen3-8B-W8A16-INT8-sym|$SHARED"
    ARMS[w8fp8]="/data/smcho/ckpts/Qwen3-8B-W8A16-FP8|$SHARED VLLM_TEST_FORCE_FP8_MARLIN=1"
    ARMS[fp8dyn]="/data/smcho/ckpts/Qwen3-8B-FP8-dynamic|$SHARED TORCHINDUCTOR_FORCE_DISABLE_CACHES=1"
    ARMS[win128]="$MODEL|$(winplain 128)"
    ARMS[win512]="$MODEL|$(winplain 512)"
    ARMS[win2048]="$MODEL|$(winplain 2048)"
    ARMS[win8192]="$MODEL|$(winplain 8192)"
    ARMS[skipb2]="$MODEL|$SHARED VLLM_SELF_SPEC_DRAFT_SKIP_LAYERS=2,8"
    ARMS[skipb4]="$MODEL|$SHARED VLLM_SELF_SPEC_DRAFT_SKIP_LAYERS=2,4,8,10"
    ARMS[kvq]="$MODEL|$KVQSTACK"
    ;;
  mla)
    MODEL="deepseek-ai/DeepSeek-V2-Lite"; TP=1; KVLIM=250000
    GPU="${STAGEA_GPU:-7}"
    ARMS[w4a16]="/data/smcho/ckpts/DeepSeek-V2-Lite-W4A16-INT4-sym|$SHARED VLLM_DISABLED_KERNELS=MarlinLinearKernel"
    ARMS[w8chan]="/data/smcho/ckpts/DeepSeek-V2-Lite-W8A16-INT8-chan|$SHARED"
    ARMS[fp8dyn]="/data/smcho/ckpts/DeepSeek-V2-Lite-FP8-dynamic|$SHARED"
    ARMS[win128]="$MODEL|$(winplain 128)"
    ARMS[win512]="$MODEL|$(winplain 512)"
    ARMS[win2048]="$MODEL|$(winplain 2048)"
    ARMS[win8192]="$MODEL|$(winplain 8192)"
    ARMS[skipb2]="$MODEL|$SHARED VLLM_SELF_SPEC_DRAFT_SKIP_LAYERS=10,11"
    ARMS[skipb4]="$MODEL|$SHARED VLLM_SELF_SPEC_DRAFT_SKIP_LAYERS=10,11,16,22"
    ARMS[kvq]="$MODEL|$KVQSTACK"
    ;;
  moe)
    MODEL="Qwen/Qwen3-30B-A3B"; TP=2; KVLIM=130000
    GPU="${STAGEA_GPU:-6,7}"
    ARMS[w4a16]="/data/smcho/ckpts/Qwen3-30B-A3B-W4A16-INT4-sym|$SHARED"
    ARMS[w8chan]="/data/smcho/ckpts/Qwen3-30B-A3B-W8A16-INT8-chan|$SHARED"
    ARMS[fp8dyn]="/data/smcho/ckpts/Qwen3-30B-A3B-FP8-dynamic|$SHARED"
    ARMS[win128]="$MODEL|$(winplain 128)"
    ARMS[win512]="$MODEL|$(winplain 512)"
    ARMS[win2048]="$MODEL|$(winplain 2048)"
    ARMS[win8192]="$MODEL|$(winplain 8192)"
    ARMS[skipb2]="$MODEL|$SHARED VLLM_SELF_SPEC_DRAFT_SKIP_LAYERS=15,23"
    ARMS[skipb4]="$MODEL|$SHARED VLLM_SELF_SPEC_DRAFT_SKIP_LAYERS=13,15,23,24"
    ARMS[kvq]="$MODEL|$KVQSTACK"
    ;;
  llama)
    MODEL="NousResearch/Meta-Llama-3.1-8B-Instruct"; TP=1; KVLIM=260000
    GPU="${STAGEA_GPU:-7}"
    SB2=$(python3 -c "import json;print(json.load(open('$PHASE/data/skipsets_llama.json'))['b2'])" 2>/dev/null || echo "2,8")
    SB4=$(python3 -c "import json;print(json.load(open('$PHASE/data/skipsets_llama.json'))['b4'])" 2>/dev/null || echo "2,4,8,10")
    ARMS[w4a16]="/data/smcho/ckpts/Llama31-8B-Instruct-W4A16-INT4-sym|$SHARED"
    ARMS[w8int8]="/data/smcho/ckpts/Llama31-8B-Instruct-W8A16-INT8-sym|$SHARED"
    ARMS[fp8dyn]="/data/smcho/ckpts/Llama31-8B-Instruct-FP8-dynamic|$SHARED"
    ARMS[win128]="$MODEL|$(winstack 128)"
    ARMS[win512]="$MODEL|$(winstack 512)"
    ARMS[win2048]="$MODEL|$(winstack 2048)"
    ARMS[win8192]="$MODEL|$(winstack 8192)"
    ARMS[skipb2]="$MODEL|$SHARED VLLM_SELF_SPEC_DRAFT_SKIP_LAYERS=$SB2"
    ARMS[skipb4]="$MODEL|$SHARED VLLM_SELF_SPEC_DRAFT_SKIP_LAYERS=$SB4"
    ARMS[kvq]="$MODEL|$KVQSTACK"
    ;;
  q3_32b)
    MODEL="Qwen/Qwen3-32B"; TP=2; KVLIM=130000
    GPU="${STAGEA_GPU:-6,7}"
    SB2=$(python3 -c "import json;print(json.load(open('$PHASE/data/skipsets_q3_32b.json'))['b2'])" 2>/dev/null || echo "2,8")
    SB4=$(python3 -c "import json;print(json.load(open('$PHASE/data/skipsets_q3_32b.json'))['b4'])" 2>/dev/null || echo "2,4,8,10")
    ARMS[w4gptq]="/data/smcho/ckpts/Qwen3-32B-W4A16-INT4-gptq|$SHARED"
    ARMS[w4a8]="/data/smcho/ckpts/Qwen3-32B-W4A8-gptq|$SHARED"
    ARMS[w4a8hum]="/data/smcho/ckpts/Qwen3-32B-W4A8-gptq|$SHARED VLLM_DISABLED_KERNELS=MacheteLinearKernel,CutlassW4A8LinearKernel,AllSparkLinearKernel"
    ARMS[w8fp8]="/data/smcho/ckpts/Qwen3-32B-W8A16-FP8|$SHARED VLLM_TEST_FORCE_FP8_MARLIN=1"
    ARMS[win128]="$MODEL|$(winstack 128)"
    ARMS[win512]="$MODEL|$(winstack 512)"
    ARMS[win2048]="$MODEL|$(winstack 2048)"
    ARMS[win8192]="$MODEL|$(winstack 8192)"
    ARMS[skipb2]="$MODEL|$SHARED VLLM_SELF_SPEC_DRAFT_SKIP_LAYERS=$SB2"
    ARMS[skipb4]="$MODEL|$SHARED VLLM_SELF_SPEC_DRAFT_SKIP_LAYERS=$SB4"
    ARMS[kvq]="$MODEL|$KVQSTACK"
    ;;
  *) echo "unknown arch $ARCH"; exit 1;;
esac

# Kill every smcho compute process on THIS run's GPUs (by nvidia-smi
# pid, not name patterns: orphaned EngineCores keep their spawn cmdline
# so pgrep -f misses them; observed 75 GiB held after a timeout kill).
gpu_cleanup() {
  local uuids
  uuids=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader |
    awk -F', ' -v g="$GPU" 'BEGIN{n=split(g,a,",");for(i=1;i<=n;i++)w[a[i]]=1} w[$1]{print $2}')
  while read -r pid uuid _; do
    pid=${pid%,}; uuid=${uuid%,}
    echo "$uuids" | grep -q "$uuid" || continue
    [ "$(ps -o user= -p "$pid" 2>/dev/null)" = "smcho" ] || continue
    kill "$pid" 2>/dev/null || true
    sleep 2
    kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null
  done < <(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | tr -d ',')
  sleep 3
}

run_one() {  # <lever> <karm> <draft> <extra-envs>
  local lever=$1 karm=$2 draft=$3 extra=$4
  local csv="cells_93_${ARCH}_${lever}.csv"
  if [ -f "$P82DATA/$csv" ] && grep -q "^${karm}," "$P82DATA/$csv"; then
    echo "[A:$ARCH] skip $lever/$karm (done)"; return 0
  fi
  echo "[A:$ARCH] measure $lever/$karm"
  env CUDA_VISIBLE_DEVICES=$GPU COMPILE_MODEL="$MODEL" \
      COMPILE_DRAFT="$draft" COMPILE_TP=$TP \
      COMPILE_BATCHES="$BATCHES" COMPILE_CTXS="$CTXS" \
      COMPILE_KV_LIMIT=$KVLIM COMPILE_CELLS="$csv" \
      COMPILE_TABLE="table_93_${ARCH}_${lever}.json" \
      $extra timeout 3600 .venv/bin/python "$COMPILE" --measure "$karm" \
      >> "$PHASE/logs/stage_a_${ARCH}.log" 2>&1 \
      || echo "[A:$ARCH] FAIL $lever/$karm (continuing)"
  gpu_cleanup
}

# AR baseline once per arch (no draft; K=0)
run_one off off "$MODEL" ""

for lever in "${!ARMS[@]}"; do
  IFS='|' read -r draft extra <<< "${ARMS[$lever]}"
  for karm in $KS; do
    run_one "$lever" "$karm" "$draft" "$extra"
  done
done

cp -f "$P82DATA"/cells_93_${ARCH}_*.csv "$PHASE/data/" 2>/dev/null
echo "[A:$ARCH] DONE; CSVs copied to $PHASE/data/"
