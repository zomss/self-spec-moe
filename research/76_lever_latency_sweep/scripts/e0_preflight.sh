#!/bin/bash
# Phase 76 E0 -- infrastructure-parity preflight (README E0 section). Run FIRST.
#
# One tiny standalone-engine run per (arm x model); asserts each lever arm rides the
# SAME infrastructure as its L0 denominator except the intended delta:
#   - attention backend (log: "Using X attention backend out of potential backends")
#   - cudagraph mode    (config dump: "cudagraph_mode': <CUDAGraphMode.X")
#   - MoE backend + prepare/finalize class (oracle info_once lines)
#   - config plumbing that logs NOTHING (sliding_window, num_hidden_layers) via
#     e0_probe_cfg.py -- the per-layer Attention debug line is conditional
#     (attention.py:271-293), so a log grep would be vacuous.
#   - LOCAL_ROUTE / SKIP_A2A engagement via the info_once markers added to
#     all2all.py (the branches were silent).
# Includes ONE deliberate trap arm (d_kvq_e5m2: fp8_e5m2 KV must FALL OFF FA3) to
# prove the checker detects backend flips, not just record them (P74 trap class).
#
# Cheap by design: input 256, 4 decodes, b1, 1 iter -- backend/kernel selection does
# not depend on ctx/batch. ~25 launches; dense ~1-2 min, TP4 ~3-6 min each.
# Run BY PATH: `bash scripts/e0_preflight.sh [dense|moe|mla|all]`
set -u
PHASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$PHASE/../.." && pwd)"
source "$PHASE/scripts/env_e76.sh"
PY="$REPO/.venv/bin/python"
mkdir -p "$PHASE/logs" "$PHASE/data"
GROUP="${1:-all}"
ME="$(whoami)"
kill_mine(){ pkill -9 -u "$ME" -f 'vllm[.]entrypoints' 2>/dev/null; pkill -9 -u "$ME" -f 'EngineCor[e]' 2>/dev/null; sleep 4; }

# arm <name> <gpus> <par> <model> <timeout_s> [extra vllm args...]
# par: tp1 | dp4 (attention-DP4+EP4, the P74/P24 MoE fabric -- pure TP4+EP has NO
# dispatch collective (NoDPEPModular, E0 finding), so the a2a path only exists
# under DP+EP). Env deltas via E0_ENV="VAR=1 VAR2=..." (applied in the subshell).
#
# tp1 runs `bench latency` (offline). dp4 MUST run `vllm serve`: the offline LLM
# API refuses internal DP (llm.py:298 "not supported for single-process usage").
# Serve-mode emits the same selection log lines; one completion proves decode runs.
E0_PORT=18076
arm(){
  local name=$1 gpus=$2 par=$3 model=$4 tmo=$5; shift 5
  local L="$PHASE/logs/e0_${name}.log"
  echo "[e0] arm $name ($(date +%H:%M:%S)) -> $(basename "$L")"
  kill_mine
  ( source "$PHASE/scripts/env_e76.sh"
    export CUDA_VISIBLE_DEVICES="$gpus"
    for kv in ${E0_ENV:-}; do export "${kv?}"; done
    case "$par" in
      tp1)
        timeout "$tmo" "$PY" -m vllm.entrypoints.cli.main bench latency \
          --model "$model" --tensor-parallel-size 1 \
          --input-len 256 --output-len 4 --batch-size 4 --max-model-len 1024 \
          --gpu-memory-utilization 0.85 --num-iters-warmup 1 --num-iters 1 \
          "$@" > "$L" 2>&1
        echo "E0EXIT=$?" > "$L.status"
        ;;
      dp4)
        timeout "$tmo" "$PY" -m vllm.entrypoints.cli.main serve "$model" \
          --data-parallel-size 4 --enable-expert-parallel \
          --max-model-len 1024 --gpu-memory-utilization 0.85 \
          --port "$E0_PORT" "$@" > "$L" 2>&1 &
        srv=$!
        ok=1; t=0
        while [ "$t" -lt "$tmo" ]; do
          if curl -sf "localhost:$E0_PORT/health" > /dev/null 2>&1; then ok=0; break; fi
          kill -0 "$srv" 2>/dev/null || break     # server died
          sleep 5; t=$((t+5))
        done
        if [ "$ok" -eq 0 ]; then
          curl -sf "localhost:$E0_PORT/v1/completions" \
            -H 'Content-Type: application/json' \
            -d "{\"model\": \"$model\", \"prompt\": \"The capital of France is\", \"max_tokens\": 8}" \
            > /dev/null 2>&1 || ok=1
        fi
        kill "$srv" 2>/dev/null; sleep 2; kill -9 "$srv" 2>/dev/null
        # Status goes to a SEPARATE file: the server holds $L open WITHOUT
        # O_APPEND, so its post-kill shutdown writes overwrite anything we
        # append to $L (E0SERVE_OK vanished this way on the first run).
        { [ "$ok" -eq 0 ] && echo "E0SERVE_OK"; echo "E0EXIT=$ok"; } > "$L.status"
        ;;
      *) echo "bad par=$par" > "$L"; echo "E0EXIT=1" > "$L.status" ;;
    esac
  )
  { grep -hE 'Avg latency' "$L"; cat "$L.status" 2>/dev/null; } | tail -2 | sed 's/^/       /'
  kill_mine
}

# probe <name> <model> [e0_probe_cfg args...]  -- config plumbing, appended to arm log
probe(){
  local name=$1 model=$2; shift 2
  local L="$PHASE/logs/e0_${name}.log"
  ( source "$PHASE/scripts/env_e76.sh"
    export CUDA_VISIBLE_DEVICES=""       # config-only; no GPU needed
    "$PY" "$PHASE/scripts/e0_probe_cfg.py" --model "$model" "$@" 2>&1 | grep -E '^E0PROBE|Error' \
      || echo "E0PROBE FAILED"
  ) >> "$L"
  grep -h '^E0PROBE' "$L" | tail -1 | sed 's/^/       /'
}

# whichkernel <name> <bits> [marlin]  -- dense MP-kernel chooser, appended to arm log
whichkernel(){
  local name=$1 bits=$2 kern=${3:-auto}
  local L="$PHASE/logs/e0_${name}.log"
  ( source "$PHASE/scripts/env_e76.sh"
    [ "$kern" = marlin ] && e76_force_marlin
    "$PY" "$PHASE/scripts/which_kernel.py" --bits "$bits" 2>/dev/null | grep '^KERNEL=' \
      | sed 's/^/E0/' || echo "E0KERNEL=PROBE-FAILED"
  ) >> "$L"
  grep -h '^E0KERNEL' "$L" | tail -1 | sed 's/^/       /'
}

DL=28; ML=48; SL=27   # num_hidden_layers: dense / moe / mla
DWIN='{"sliding_window": 512, "use_sliding_window": true, "max_window_layers": 28}'
MWIN='{"sliding_window": 512, "use_sliding_window": true, "max_window_layers": 48}'
SWIN='{"sliding_window": 512}'

if [ "$GROUP" = all ] || [ "$GROUP" = dense ]; then
  echo "[e0] === DENSE ($E76_DENSE, TP1, GPU $E76_GPU) ==="
  arm d_bf16      "$E76_GPU" tp1 "$E76_DENSE" 600
  if [ -e "$E76_W4" ]; then
    E0_ENV="" arm d_w4auto   "$E76_GPU" tp1 "$E76_W4" 600
    whichkernel d_w4auto 4 auto
    E0_ENV="VLLM_DISABLED_KERNELS=MacheteLinearKernel,CutlassW4A8LinearKernel,AllSparkLinearKernel" \
      arm d_w4marlin "$E76_GPU" tp1 "$E76_W4" 600
    whichkernel d_w4marlin 4 marlin
  else
    echo "[e0] SKIP d_w4auto/d_w4marlin -- checkpoint missing: $E76_W4 (P75 make_ckpts.sh)"
  fi
  arm d_fp8w8a8   "$E76_GPU" tp1 "$E76_DENSE" 600 --quantization fp8
  arm d_kvq       "$E76_GPU" tp1 "$E76_DENSE" 600 --kv-cache-dtype fp8_e4m3
  arm d_kvq_e5m2  "$E76_GPU" tp1 "$E76_DENSE" 600 --kv-cache-dtype fp8_e5m2   # TRAP CONTROL
  arm d_win       "$E76_GPU" tp1 "$E76_DENSE" 600 --hf-overrides "$DWIN"
  probe d_win "$E76_DENSE" --hf-overrides "$DWIN" --max-model-len 1024
  arm d_bf16dummy "$E76_GPU" tp1 "$E76_DENSE" 600 --load-format dummy
  arm d_skip50    "$E76_GPU" tp1 "$E76_DENSE" 600 --load-format dummy \
      --hf-overrides "{\"num_hidden_layers\": $((DL/2))}"
  probe d_skip50 "$E76_DENSE" --hf-overrides "{\"num_hidden_layers\": $((DL/2))}"
fi

if [ "$GROUP" = all ] || [ "$GROUP" = moe ]; then
  echo "[e0] === MoE GQA ($E76_MOE, DP4+EP4, GPUs $E76_MOE_GPUS) ==="
  arm m_bf16      "$E76_MOE_GPUS" dp4 "$E76_MOE" 1200
  arm m_fp8marlin "$E76_MOE_GPUS" dp4 "$E76_MOE" 1200 --quantization fp8 --moe-backend marlin
  arm m_fp8block  "$E76_MOE_GPUS" dp4 "$E76_MOE" 1200 --quantization fp8_per_block
  arm m_kvq       "$E76_MOE_GPUS" dp4 "$E76_MOE" 1200 --kv-cache-dtype fp8_e4m3
  arm m_win       "$E76_MOE_GPUS" dp4 "$E76_MOE" 1200 --hf-overrides "$MWIN"
  probe m_win "$E76_MOE" --hf-overrides "$MWIN" --max-model-len 1024
  arm m_bf16dummy "$E76_MOE_GPUS" dp4 "$E76_MOE" 1200 --load-format dummy
  arm m_skip50    "$E76_MOE_GPUS" dp4 "$E76_MOE" 1200 --load-format dummy \
      --hf-overrides "{\"num_hidden_layers\": $((ML/2))}"
  probe m_skip50 "$E76_MOE" --hf-overrides "{\"num_hidden_layers\": $((ML/2))}"
  E0_ENV="VLLM_SELF_SPEC_LOCAL_ROUTE=1" arm m_localroute "$E76_MOE_GPUS" dp4 "$E76_MOE" 1200
  E0_ENV="VLLM_SELF_SPEC_SKIP_A2A=1"    arm m_skipa2a    "$E76_MOE_GPUS" dp4 "$E76_MOE" 1200
fi

if [ "$GROUP" = all ] || [ "$GROUP" = mla ]; then
  echo "[e0] === MoE MLA ($E76_MLA, DP4+EP4, GPUs $E76_MOE_GPUS) ==="
  arm ds_bf16      "$E76_MOE_GPUS" dp4 "$E76_MLA" 1200 --trust-remote-code
  arm ds_fp8marlin "$E76_MOE_GPUS" dp4 "$E76_MLA" 1200 --trust-remote-code \
      --quantization fp8 --moe-backend marlin
  arm ds_fp8block  "$E76_MOE_GPUS" dp4 "$E76_MLA" 1200 --trust-remote-code \
      --quantization fp8_per_block
  arm ds_kvq       "$E76_MOE_GPUS" dp4 "$E76_MLA" 1200 --trust-remote-code \
      --kv-cache-dtype fp8_e4m3
  arm ds_win       "$E76_MOE_GPUS" dp4 "$E76_MLA" 1200 --trust-remote-code \
      --hf-overrides "$SWIN"
  probe ds_win "$E76_MLA" --trust-remote-code --hf-overrides "$SWIN" --max-model-len 1024
  arm ds_bf16dummy "$E76_MOE_GPUS" dp4 "$E76_MLA" 1200 --trust-remote-code --load-format dummy
  arm ds_skip50    "$E76_MOE_GPUS" dp4 "$E76_MLA" 1200 --trust-remote-code --load-format dummy \
      --hf-overrides "{\"num_hidden_layers\": $(((SL+1)/2))}"
  probe ds_skip50 "$E76_MLA" --trust-remote-code \
      --hf-overrides "{\"num_hidden_layers\": $(((SL+1)/2))}"
fi

kill_mine
echo "[e0] launches done ($(date +%H:%M:%S)); building assertion table"
"$PY" "$PHASE/scripts/e0_assert.py" --logs "$PHASE/logs" --out "$PHASE/data/e0_table.md" --group "$GROUP"
