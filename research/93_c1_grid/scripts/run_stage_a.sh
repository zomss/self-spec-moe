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

declare -A ARMS
case "$ARCH" in
  dense)
    MODEL="Qwen/Qwen3-8B"; TP=1; KVLIM=260000
    GPU="${STAGEA_GPU:-6}"
    ARMS[w4a16]="$HOME/ckpts/Qwen3-8B-W4A16-INT4|"
    ARMS[w4a8cut]="$HOME/ckpts/Qwen3-8B-W4A8-gptq|"
    ARMS[w4a8hum]="$HOME/ckpts/Qwen3-8B-W4A8-gptq|VLLM_DISABLED_KERNELS=MacheteLinearKernel,CutlassW4A8LinearKernel,AllSparkLinearKernel"
    ARMS[w8int8]="$HOME/ckpts/Qwen3-8B-W8A16-INT8-sym|"
    ARMS[w8fp8]="$HOME/ckpts/Qwen3-8B-W8A16-FP8|VLLM_TEST_FORCE_FP8_MARLIN=1"
    ARMS[fp8dyn]="$HOME/ckpts/Qwen3-8B-FP8-dynamic|"
    ARMS[win128]="$MODEL|VLLM_SELF_SPEC_DRAFT_KV_WINDOW=128"
    ARMS[win512]="$MODEL|VLLM_SELF_SPEC_DRAFT_KV_WINDOW=512"
    ARMS[win2048]="$MODEL|VLLM_SELF_SPEC_DRAFT_KV_WINDOW=2048"
    ARMS[win8192]="$MODEL|VLLM_SELF_SPEC_DRAFT_KV_WINDOW=8192"
    ARMS[skipb2]="$MODEL|VLLM_SELF_SPEC_DRAFT_SKIP_LAYERS=2,8"
    ARMS[skipb4]="$MODEL|VLLM_SELF_SPEC_DRAFT_SKIP_LAYERS=2,4,8,10"
    ARMS[kvq]="$MODEL|VLLM_SELF_SPEC_DRAFT_KV_DTYPE=fp8"
    ;;
  mla)
    MODEL="deepseek-ai/DeepSeek-V2-Lite"; TP=1; KVLIM=250000
    GPU="${STAGEA_GPU:-7}"
    ARMS[w4a16]="$HOME/ckpts/DeepSeek-V2-Lite-W4A16-INT4-sym|"
    ARMS[w8chan]="$HOME/ckpts/DeepSeek-V2-Lite-W8A16-INT8-chan|"
    ARMS[fp8dyn]="$HOME/ckpts/DeepSeek-V2-Lite-FP8-dynamic|"
    ARMS[win128]="$MODEL|VLLM_SELF_SPEC_DRAFT_KV_WINDOW=128"
    ARMS[win512]="$MODEL|VLLM_SELF_SPEC_DRAFT_KV_WINDOW=512"
    ARMS[win2048]="$MODEL|VLLM_SELF_SPEC_DRAFT_KV_WINDOW=2048"
    ARMS[win8192]="$MODEL|VLLM_SELF_SPEC_DRAFT_KV_WINDOW=8192"
    ARMS[skipb2]="$MODEL|VLLM_SELF_SPEC_DRAFT_SKIP_LAYERS=10,11"
    ARMS[skipb4]="$MODEL|VLLM_SELF_SPEC_DRAFT_SKIP_LAYERS=10,11,16,22"
    ARMS[kvq]="$MODEL|VLLM_SELF_SPEC_DRAFT_KV_DTYPE=fp8"
    ;;
  moe)
    MODEL="Qwen/Qwen3-30B-A3B"; TP=2; KVLIM=130000
    GPU="${STAGEA_GPU:-6,7}"
    ARMS[w4a16]="$HOME/ckpts/Qwen3-30B-A3B-W4A16-INT4-sym|"
    ARMS[w8chan]="$HOME/ckpts/Qwen3-30B-A3B-W8A16-INT8-chan|"
    ARMS[fp8dyn]="$HOME/ckpts/Qwen3-30B-A3B-FP8-dynamic|"
    ARMS[win128]="$MODEL|VLLM_SELF_SPEC_DRAFT_KV_WINDOW=128"
    ARMS[win512]="$MODEL|VLLM_SELF_SPEC_DRAFT_KV_WINDOW=512"
    ARMS[win2048]="$MODEL|VLLM_SELF_SPEC_DRAFT_KV_WINDOW=2048"
    ARMS[win8192]="$MODEL|VLLM_SELF_SPEC_DRAFT_KV_WINDOW=8192"
    ARMS[skipb2]="$MODEL|VLLM_SELF_SPEC_DRAFT_SKIP_LAYERS=15,23"
    ARMS[skipb4]="$MODEL|VLLM_SELF_SPEC_DRAFT_SKIP_LAYERS=13,15,23,24"
    ARMS[kvq]="$MODEL|VLLM_SELF_SPEC_DRAFT_KV_DTYPE=fp8"
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
