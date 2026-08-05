#!/usr/bin/env bash
# W4a -- transition cost + parked cost, under the W3 protocol (notune,
# ITERS=4, GATE_DEBUG for exact duty/flip counts). Supersedes 95/e1p's
# lottery-era plan; same two-point duty design.
#
#   arm      table              probes      measures
#   off      (no spec)          --          AR anchor
#   parked   park-everywhere    burst 0     parked-engine cost (W4 item 2)
#   dutyA    normal w512        2/64        3.1% duty, ~32 cycles/2048 steps
#   dutyB    normal w512        8/256       3.1% duty,  ~8 cycles/2048 steps
#
# Cell: llama w512, regimes R8 (policy parks -> probes are pure forays)
# + R4 (control: policy arms and wins; probes must not degrade it).
# If loss(dutyA) ~ 4x loss(dutyB) vs parked -> cost is TRANSITIONS;
# if loss(dutyA) ~ loss(dutyB) -> cost is DUTY. kpick lines in the logs
# give the realized flip counts, so the per-flip cost is computed from
# MEASURED flips, not the nominal schedule.
#
# Pre-registered predictions (results_w1/w2 + phase-95 estimate):
#   P-W4a1  parked - off is -1..-3% at b16 (phase-95: -2.5%), ~0% at b8
#   P-W4a2  the loss scales with flips; per-cycle cost ~20 ms
set -uo pipefail
cd /data/smcho/self-spec-moe

PHASE=research/96_selector_foundations
P95=research/95_c3_deploy
LOG="$PHASE/logs"; mkdir -p "$LOG" "$PHASE/data/w4"

export PATH="/data/smcho/self-spec-moe/.venv/bin:$PATH"
export HF_HOME=/data/smcho/huggingface
export VLLM_CACHE_ROOT=/data/smcho/.cache/vllm
export TRITON_CACHE_DIR=/data/smcho/.cache/triton
export TMPDIR=/data/smcho/tmp

export VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1
export VLLM_SELF_SPEC_CPU_ORCH=1 VLLM_SELF_SPEC_SHARED_KV=1
export VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE=1 VLLM_SELF_SPEC_SKIP_PREFILL_DRAFT=1
export VLLM_SELF_SPEC_DRAFT_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1
export VLLM_SELF_SPEC_GATE_DEBUG=1

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

run_arm() {  # gpu name window burst interval policy
  local gpu="$1" name="$2" window="$3" burst="$4" interval="$5" policy="$6"
  local out="$PHASE/data/w4/w4a_${name}.json"
  if [ -f "$out" ]; then echo "[W4a] skip $name (exists)"; return; fi
  echo "[W4a] === gpu=$gpu $name ==="
  cleanup "$gpu"
  env CUDA_VISIBLE_DEVICES="$gpu" \
      VLLM_SELF_SPEC_ACCEPT_PROBE_BURST="$burst" \
      VLLM_SELF_SPEC_ACCEPT_PROBE_INTERVAL="$interval" \
      E95_POLICY="$policy" \
      E95_ARCH=llama E95_WINDOW="$window" E95_SEED=0 E95_REGIMES=R8,R4 \
      E95_ITERS=4 E95_TUNE=0 E95_OUT="$out" \
      timeout 3600 .venv/bin/python "$P95/scripts/run_e0_envelope.py" \
        >> "$LOG/w4a_${name}.log" 2>&1
  [ $? -ne 0 ] && echo "[W4a] FAILED $name (see $LOG/w4a_${name}.log)"
}

PARK="$PHASE/data/w4/policy_llama_park_everywhere_w512.json"

( run_arm 0 off    off 2 64  ""
  run_arm 0 parked 512 0 64  "$PARK" ) &
L0=$!
( run_arm 1 dutyA  512 2 64  ""
  run_arm 1 dutyB  512 8 256 "" ) &
L1=$!
wait $L0 $L1
cleanup 0; cleanup 1
echo "[W4a] matrix complete"
ls -la "$PHASE/data/w4/"
