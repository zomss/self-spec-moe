#!/bin/bash
# E4 -- the MLA crossover. Does shrinking KV (MLA) flip weight-only quant from parity
# (GQA MoE, P74) to a WIN, isolating KV-size as the crossover knob?
#
# Metric: weight-only-fp8 draft/verify cost ratio = decode-step(fp8-marlin)/decode-step(bf16),
# measured by the two-output-length SLOPE (prefill cancels). fp8+marlin = pure read cut
# (dequant, no compute cut). At 16k b8 the GQA MoE is KV-bound (KV 12.9 GB >> weight) but
# the MLA MoE is not (KV 4 GB). PREDICT: DeepSeek(MLA) ratio << Qwen3(GQA) ratio (~1.0, P74).
#
# EP4 on GPU 2-5. PATH includes .venv/bin so the MLA JIT step finds `ninja`. Run BY PATH.
set -u
PHASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$PHASE/../.." && pwd)"
export PATH="$REPO/.venv/bin:$PATH"          # MLA JIT needs ninja
PY="$REPO/.venv/bin/python"
export CUDA_VISIBLE_DEVICES=2,3,4,5
export HF_HUB_OFFLINE=1 VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
CTX=16384; B="${E4_B:-8}"; O1=8; O2=40         # slope over 32 decode steps; E4_B=1 for mem-bound
mkdir -p "$PHASE/logs" "$PHASE/data"
kill_mine(){ pkill -9 -u "$(whoami)" -f 'vllm[.]entrypoints' 2>/dev/null; pkill -9 -u "$(whoami)" -f 'EngineCor[e]' 2>/dev/null; sleep 4; }

arm(){  # name model quantflag
  local name=$1 model=$2 qflag=$3
  for O in $O1 $O2; do
    local J="$PHASE/data/e4_${name}_b${B}_o${O}.json"
    local L="$PHASE/logs/e4_${name}_b${B}_o${O}.log"
    kill_mine
    timeout 900 "$PY" -m vllm.entrypoints.cli.main bench latency \
      --model "$model" --trust-remote-code \
      --tensor-parallel-size 4 --enable-expert-parallel $qflag \
      --input-len "$CTX" --output-len "$O" --batch-size "$B" \
      --max-model-len $((CTX+512)) --gpu-memory-utilization 0.85 \
      --num-iters-warmup 1 --num-iters 3 --output-json "$J" > "$L" 2>&1
    echo "[e4] ${name} o=${O}: $(grep -hiE 'Avg latency' "$L" | tail -1)"
    grep -hiE 'Fp8 MoE backend|FLASH_ATTN_MLA|Error|Traceback|out of memory|ninja' "$L" | tail -1 | sed 's/^/       /'
  done
}

echo "[e4] START ($(date +%H:%M:%S)) ctx=$CTX b=$B slope=$((O2-O1))"
arm ds_bf16   deepseek-ai/DeepSeek-V2-Lite ""
arm ds_fp8m   deepseek-ai/DeepSeek-V2-Lite "--quantization fp8 --moe-backend marlin"
arm qw_bf16   Qwen/Qwen3-30B-A3B ""
arm qw_fp8m   Qwen/Qwen3-30B-A3B "--quantization fp8 --moe-backend marlin"
kill_mine

echo "[e4] === ratios (decode-step fp8-marlin / bf16 = weight-only-fp8 Tq/Tp) ==="
"$PY" - "$PHASE/data" $O1 $O2 $B <<'PY'
import json,sys,os
d,o1,o2,B=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),sys.argv[4]
def step(name):
    try:
        t1=json.load(open(f"{d}/e4_{name}_b{B}_o{o1}.json"))["avg_latency"]
        t2=json.load(open(f"{d}/e4_{name}_b{B}_o{o2}.json"))["avg_latency"]
        return (t2-t1)/(o2-o1)*1e3
    except Exception as e: return None
for tag,bf,fp in [("DeepSeek-V2-Lite (MLA)","ds_bf16","ds_fp8m"),("Qwen3-30B-A3B (GQA)","qw_bf16","qw_fp8m")]:
    b=step(bf); f=step(fp)
    if b and f: print(f"  {tag:26s} bf16={b:.2f}ms  fp8m={f:.2f}ms  ratio={f/b:.3f}")
    else:       print(f"  {tag:26s} MISSING (bf16={b} fp8m={f})")
print("\n  PREDICT: DeepSeek(MLA) ratio < Qwen3(GQA) ratio (~1.0). ratio<1 => weight-quant PAYS on the MLA MoE.")
PY
echo "[e4] DONE ($(date +%H:%M:%S))"
