#!/bin/bash
# One-off: rerun ONLY the two fp8block arms after the nvcc-not-found fix, then
# rebuild both group tables. Reuses the exact arm semantics of e0_preflight.sh.
set -u
PHASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$PHASE/../.." && pwd)"
source "$PHASE/scripts/env_e76.sh"
PY="$REPO/.venv/bin/python"
ME="$(whoami)"
kill_mine(){ pkill -9 -u "$ME" -f 'vllm[.]entrypoints' 2>/dev/null; pkill -9 -u "$ME" -f 'EngineCor[e]' 2>/dev/null; sleep 4; }
E0_PORT=18076
serve_arm(){  # name model extra...
  local name=$1 model=$2; shift 2
  local L="$PHASE/logs/e0_${name}.log"
  echo "[e0] arm $name ($(date +%H:%M:%S))"
  kill_mine
  ( source "$PHASE/scripts/env_e76.sh"
    export CUDA_VISIBLE_DEVICES="$E76_MOE_GPUS"
    timeout 1200 "$PY" -m vllm.entrypoints.cli.main serve "$model" \
      --data-parallel-size 4 --enable-expert-parallel \
      --max-model-len 1024 --gpu-memory-utilization 0.85 \
      --port "$E0_PORT" "$@" > "$L" 2>&1 &
    srv=$!; ok=1; t=0
    while [ "$t" -lt 1200 ]; do
      curl -sf "localhost:$E0_PORT/health" > /dev/null 2>&1 && { ok=0; break; }
      kill -0 "$srv" 2>/dev/null || break
      sleep 5; t=$((t+5))
    done
    if [ "$ok" -eq 0 ]; then
      curl -sf "localhost:$E0_PORT/v1/completions" -H 'Content-Type: application/json' \
        -d "{\"model\": \"$model\", \"prompt\": \"The capital of France is\", \"max_tokens\": 8}" \
        > /dev/null 2>&1 || ok=1
    fi
    kill "$srv" 2>/dev/null; sleep 2; kill -9 "$srv" 2>/dev/null
    { [ "$ok" -eq 0 ] && echo "E0SERVE_OK"; echo "E0EXIT=$ok"; } > "$L.status"
  )
  cat "$L.status" | sed 's/^/       /'
  kill_mine
}
serve_arm m_fp8block  Qwen/Qwen3-30B-A3B --quantization fp8_per_block
serve_arm ds_fp8block deepseek-ai/DeepSeek-V2-Lite --trust-remote-code --quantization fp8_per_block
"$PY" "$PHASE/scripts/e0_assert.py" --logs "$PHASE/logs" --out "$PHASE/data/e0_table_moe.md" --group moe
"$PY" "$PHASE/scripts/e0_assert.py" --logs "$PHASE/logs" --out "$PHASE/data/e0_table_mla.md" --group mla
