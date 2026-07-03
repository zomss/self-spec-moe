#!/usr/bin/env bash
# Phase 39 control: CFG=C (EP-full draft = real all-to-all = the verify) at
# temperature, to isolate whether the temperature accept-DROP is the comm-free
# divergence (CFG A) or a generic probabilistic-spec artifact. CFG C draft does
# the SAME all-to-all as the verify -> p_draft == p_target -> ratio ~1 -> should
# stay near-perfect accept under temperature. Serial, own-worker teardown.
set -u
cd /data/smcho/self-spec-moe

PY=/data/smcho/self-spec-moe/.venv/bin/python
SCRIPT=research/39_temperature_recovery/scripts/temp_sweep.py
LOGDIR=research/39_temperature_recovery/logs
mkdir -p "$LOGDIR"

export NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=1
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
export VLLM_USE_V1=1

teardown() {
  pkill -9 -u "$(id -u)" -f "temp_sweep.py" 2>/dev/null
  pkill -9 -u "$(id -u)" -f "VLLM::EngineCore" 2>/dev/null
  pkill -9 -u "$(id -u)" -f "from multiprocessing.spawn" 2>/dev/null
  sleep 6
}

run_one() {
  local cfg="$1"; local temp="$2"
  local name="sweep_cfg${cfg}_t${temp//./p}_prob_dp8_K4"
  echo "=== RUN $name ($(date +%H:%M:%S)) ==="
  E_TEMP="$temp" E_CFG="$cfg" E_PROBABILISTIC=1 E_DP=8 E_K=4 E_TAG=sweep \
    timeout 2600 "$PY" "$SCRIPT" >"$LOGDIR/${name}.log" 2>&1
  local rc=$?
  grep -h "^\[CFG\]" "$LOGDIR/${name}.log" || true
  grep -h "^RESULT" "$LOGDIR/${name}.log" || echo "  (no RESULT line; rc=$rc)"
  teardown
}

teardown
# EP-full draft control at greedy + two temps (cheap: 3 runs).
run_one C 0.0
run_one C 0.7
run_one C 1.0
echo "=== CONTROL DONE ($(date +%H:%M:%S)) ==="
