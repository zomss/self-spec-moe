#!/bin/bash
# Phase 76 E1 -- the lever x regime draft-latency sweep (README "Matrix").
#
# Metric: R(lever, cell) = TPOT(arm) / TPOT(L0-denominator), TPOT = mean time
# per output token from `vllm bench serve` (= decode-step time; TTFT/prefill
# excluded by construction). ONE server launch per arm serves ALL (batch x ctx)
# cells -- vs 2 engine relaunches per cell for the P75 slope method. Method is
# uniform across dense (TP1) and MoE/MLA (attention-DP4+EP4, serve-mode because
# offline LLM refuses internal DP -- E0 finding 1). Cross-check cell: dense
# bf16 b1/2k TPOT must ~= P75-E1's slope (5.86 ms bf16) -- asserted by analyzer.
#
# Groups (E0-validated arms; kernel-path map in results_e0.md):
#   dense: Qwen2.5-7B TP1 on GPU $E76_GPU          cells B{1,8,32} x C{2k,16k,32k}
#   moe:   Qwen3-30B-A3B DP4+EP4 on $E76_MOE_GPUS  cells B{4,8,32} x C{2k,16k,32k}
#   mla:   DeepSeek-V2-Lite DP4+EP4                cells B{4,8,32} x C{2k,16k,32k}
#   pcie:  moe arms under forced PCIe-SHM (P24 3e) cells {8,32}x16k + 32x32k
# Special pairs: L3 skip arms are dummy-load and ratio against L0dummy;
# MLA L1c pins VLLM_ATTENTION_BACKEND=FLASHMLA on BOTH arms (E0 finding 2).
#
# Usage: bash scripts/e1_sweep.sh <dense|moe|mla|pcie> [arm-name-filter-regex]
#   E1_RUNS=2 (repeats/cell)  E1_OUT=128 (decode len)  E1_PORT=18076
set -u
PHASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$PHASE/../.." && pwd)"
source "$PHASE/scripts/env_e76.sh"
PY="$REPO/.venv/bin/python"
GROUP="${1:?usage: e1_sweep.sh <dense|moe|mla|pcie> [arm-filter]}"
FILTER="${2:-.}"
RUNS="${E1_RUNS:-2}"
OUT="${E1_OUT:-128}"
PORT="${E1_PORT:-18076}"
MAXLEN=33792                      # 32k ctx + 128 out + margin; dense overrides to
                                  # 32768 (Qwen2.5 max_position_embeddings cap) and
                                  # its 32k cell uses input 32512 so ctx+out fits
D="$PHASE/data/e1"; mkdir -p "$D" "$PHASE/logs"
ME="$(whoami)"
kill_mine(){ pkill -9 -u "$ME" -f 'vllm[.]entrypoints' 2>/dev/null; pkill -9 -u "$ME" -f 'EngineCor[e]' 2>/dev/null; sleep 4; }

CELLS_B="4 8 32"; CELLS_C="2048 16384 32768"    # per-group defaults, set below
DWIN='{"sliding_window": 512, "use_sliding_window": true, "max_window_layers": 28}'
MWIN='{"sliding_window": 512, "use_sliding_window": true, "max_window_layers": 48}'
SWIN='{"sliding_window": 512}'

# run_arm <name> <par:tp1|dp4> <model> [serve extra args...]
#   env deltas via A_ENV="K=V ..."; per-arm cell override via A_CELLS="b:c b:c ..."
run_arm(){
  local name=$1 par=$2 model=$3; shift 3
  echo "$name" | grep -qE "$FILTER" || return 0
  local SL="$PHASE/logs/e1_${name}_server.log"
  local META="$D/${GROUP}_${name}.meta"
  echo "[e1] === arm $name ($(date +%H:%M:%S)) ==="
  kill_mine
  ( source "$PHASE/scripts/env_e76.sh"
    for kv in ${A_ENV:-}; do export "${kv?}"; done
    if [ "$par" = tp1 ]; then
      export CUDA_VISIBLE_DEVICES="$E76_GPU"; pargs="--tensor-parallel-size 1"
    else
      export CUDA_VISIBLE_DEVICES="$E76_MOE_GPUS"
      pargs="--data-parallel-size 4 --enable-expert-parallel"
    fi
    "$PY" -m vllm.entrypoints.cli.main serve "$model" $pargs \
      --max-model-len "$MAXLEN" --gpu-memory-utilization 0.85 \
      --port "$PORT" "$@" > "$SL" 2>&1 &
    srv=$!; t=0
    until curl -sf "localhost:$PORT/health" > /dev/null 2>&1; do
      kill -0 "$srv" 2>/dev/null || { echo "[e1] $name: server DIED (see $SL)"; exit 1; }
      [ "$t" -ge 1800 ] && { echo "[e1] $name: server TIMEOUT"; kill -9 "$srv"; exit 1; }
      sleep 5; t=$((t+5))
    done
    # provenance sidecar: backend identity + engagement markers (E0 parity, live)
    { grep -hoE "Using \S+ attention backend" "$SL" | head -1
      grep -hoE "Using \S+ (Fp8|Unquantized) MoE backend" "$SL" | head -1
      grep -hoE "Using \S*Prepare\S*" "$SL" | head -1
      grep -hoE "SELF_SPEC \S+ engaged[^\"]*" "$SL" | head -1
      grep -hoE "via SHM/direct" "$SL" | head -1; } > "$META"
    sed 's/^/       /' "$META"
    if [ "${E1_REQUIRE_PCIE:-0}" = 1 ] && ! grep -q "via SHM/direct" "$META"; then
      echo "[e1] $name: ABORT -- PCIe-SHM not engaged (no 'via SHM/direct' in NCCL log)"
      kill -9 "$srv"; exit 1
    fi
    bench(){  # <B> <CTX> <tag>
      timeout 900 "$PY" -m vllm.entrypoints.cli.main bench serve \
        --port "$PORT" --model "$model" --tokenizer "$model" \
        --dataset-name random --random-input-len "$2" --random-output-len "$OUT" \
        --ignore-eos --num-prompts "$1" --max-concurrency "$1" \
        ${3:+--save-result --result-dir "$D" --result-filename "$3"} \
        > /dev/null 2>&1
    }
    bench 4 256 ""                          # warmup: cudagraph/autotune paths
    for cell in ${A_CELLS:-${E1_CELLS:-$(for b in $CELLS_B; do for c in $CELLS_C; do echo "$b:$c"; done; done)}}; do
      b="${cell%%:*}"; c="${cell##*:}"
      l0=$(wc -l < "$SL")
      # Per-cell WARM pass (unsaved): bench serve reuses the same seed/prompts,
      # so this fills the prefix cache and the measured runs start decode-clean.
      # Without it, run 1's TPOT is inflated ~2x at b>=8 by chunked-prefill
      # interleaving (observed dense b32/16k: 57ms warm-less vs 12ms warm).
      bench "$b" "$c" ""
      for r in $(seq 1 "$RUNS"); do
        f="${GROUP}_${name}_b${b}_c$(((c+512)/1024))k_r${r}.json"
        if bench "$b" "$c" "$f"; then
          tp=$("$PY" -c "import json;print(round(json.load(open('$D/$f'))['mean_tpot_ms'],3))" 2>/dev/null)
          echo "       b=$b ctx=$c r=$r tpot=${tp}ms"
        else
          echo "       b=$b ctx=$c r=$r FAILED"
        fi
      done
      # Over-capacity detection: vLLM v1 does NOT log 'preempt'; the signature
      # is the periodic stats line showing queued requests ("Waiting: N reqs")
      # while the cell wants all B running -- the cell then measures a lower
      # effective concurrency in shifts and is invalid as a B-cell.
      if tail -n +"$((l0+1))" "$SL" | grep -qE "Waiting: [1-9][0-9]* reqs|preempt"; then
        echo "OVERCAP b${b}_c$(((c+512)/1024))k 1" >> "$META"
        echo "       b=$b ctx=$c OVER-CAPACITY (queued reqs) -- cell excluded from ratios"
      fi
    done
    kill "$srv" 2>/dev/null; sleep 2; kill -9 "$srv" 2>/dev/null
  )
  kill_mine
}

case "$GROUP" in
dense)
  CELLS_B="1 8 32"; CELLS_C="2048 16384 32512"; MAXLEN=32768
  run_arm d_bf16      tp1 "$E76_DENSE"
  if [ -e "$E76_W4" ]; then
    run_arm d_w4machete tp1 "$E76_W4"
    A_ENV="VLLM_DISABLED_KERNELS=MacheteLinearKernel,CutlassW4A8LinearKernel,AllSparkLinearKernel" \
      run_arm d_w4marlin tp1 "$E76_W4"
  else
    echo "[e1] SKIP W4 arms -- $E76_W4 missing"
  fi
  run_arm d_fp8w8a8   tp1 "$E76_DENSE" --quantization fp8
  run_arm d_kvq       tp1 "$E76_DENSE" --kv-cache-dtype fp8_e4m3
  run_arm d_win       tp1 "$E76_DENSE" --hf-overrides "$DWIN"
  run_arm d_bf16dummy tp1 "$E76_DENSE" --load-format dummy
  run_arm d_skip50    tp1 "$E76_DENSE" --load-format dummy --hf-overrides '{"num_hidden_layers": 14}'
  A_CELLS="1:2048 1:32512 32:2048 32:32512" \
    run_arm d_skip25  tp1 "$E76_DENSE" --load-format dummy --hf-overrides '{"num_hidden_layers": 21}'
  ;;
moe)
  run_arm m_bf16      dp4 "$E76_MOE"
  run_arm m_fp8marlin dp4 "$E76_MOE" --quantization fp8 --moe-backend marlin
  run_arm m_fp8block  dp4 "$E76_MOE" --quantization fp8_per_block
  run_arm m_kvq       dp4 "$E76_MOE" --kv-cache-dtype fp8_e4m3
  run_arm m_win       dp4 "$E76_MOE" --hf-overrides "$MWIN"
  run_arm m_bf16dummy dp4 "$E76_MOE" --load-format dummy
  run_arm m_skip50    dp4 "$E76_MOE" --load-format dummy --hf-overrides '{"num_hidden_layers": 24}'
  A_ENV="VLLM_SELF_SPEC_LOCAL_ROUTE=1" run_arm m_localroute dp4 "$E76_MOE"
  A_ENV="VLLM_SELF_SPEC_SKIP_A2A=1"    run_arm m_skipa2a    dp4 "$E76_MOE"
  ;;
mla)
  run_arm ds_bf16      dp4 "$E76_MLA" --trust-remote-code
  run_arm ds_fp8marlin dp4 "$E76_MLA" --trust-remote-code --quantization fp8 --moe-backend marlin
  run_arm ds_fp8block  dp4 "$E76_MLA" --trust-remote-code --quantization fp8_per_block
  run_arm ds_win       dp4 "$E76_MLA" --trust-remote-code --hf-overrides "$SWIN"
  run_arm ds_bf16dummy dp4 "$E76_MLA" --trust-remote-code --load-format dummy
  run_arm ds_skip50    dp4 "$E76_MLA" --trust-remote-code --load-format dummy --hf-overrides '{"num_hidden_layers": 14}'
  # L1c pair: BOTH arms pinned to FLASHMLA (fp8-KV flips the backend, E0 finding 2)
  A_ENV="VLLM_ATTENTION_BACKEND=FLASHMLA" run_arm ds_bf16fmla dp4 "$E76_MLA" --trust-remote-code
  A_ENV="VLLM_ATTENTION_BACKEND=FLASHMLA" run_arm ds_kvq      dp4 "$E76_MLA" --trust-remote-code --kv-cache-dtype fp8_e4m3
  ;;
pcie)
  # Forced PCIe-SHM (P24 3e): SHM/direct channels, no NVLink/NVLS/IB. Comm-bound tier.
  PCIE_ENV="NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=1 NCCL_DEBUG=INFO"
  export E1_REQUIRE_PCIE=1
  A_CELLS_PCIE="8:16384 32:16384 32:32768"
  A_ENV="$PCIE_ENV" A_CELLS="$A_CELLS_PCIE" run_arm p_bf16       dp4 "$E76_MOE"
  A_ENV="$PCIE_ENV VLLM_SELF_SPEC_LOCAL_ROUTE=1" A_CELLS="$A_CELLS_PCIE" run_arm p_localroute dp4 "$E76_MOE"
  A_ENV="$PCIE_ENV VLLM_SELF_SPEC_SKIP_A2A=1"    A_CELLS="$A_CELLS_PCIE" run_arm p_skipa2a    dp4 "$E76_MOE"
  A_ENV="$PCIE_ENV" A_CELLS="$A_CELLS_PCIE" run_arm p_fp8marlin  dp4 "$E76_MOE" --quantization fp8 --moe-backend marlin
  A_ENV="$PCIE_ENV" A_CELLS="$A_CELLS_PCIE" run_arm p_fp8block   dp4 "$E76_MOE" --quantization fp8_per_block
  A_ENV="$PCIE_ENV" A_CELLS="$A_CELLS_PCIE" run_arm p_kvq        dp4 "$E76_MOE" --kv-cache-dtype fp8_e4m3
  A_ENV="$PCIE_ENV" A_CELLS="$A_CELLS_PCIE" run_arm p_win        dp4 "$E76_MOE" --hf-overrides "$MWIN"
  ;;
*) echo "unknown group: $GROUP"; exit 1 ;;
esac

kill_mine
echo "[e1] group $GROUP done ($(date +%H:%M:%S)); analyze with:"
echo "     $PY $PHASE/scripts/e1_analyze.py --data $D"
