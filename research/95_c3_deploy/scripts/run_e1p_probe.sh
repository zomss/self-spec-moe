#!/usr/bin/env bash
# E1' step 1 -- does the probe cost scale with ARMED DUTY or with TRANSITION
# COUNT? Two arms at IDENTICAL duty and 4x different flip counts.
#
#   arm     burst/interval   armed duty   arm/disarm cycles per 2048 steps
#   dutyA   2/64             3.1%         ~32
#   dutyB   8/256            3.1%         ~8
#   parked  0/-              ~0%          ~0   (reference, E0 measured 2103.6)
#
# If the loss scales with TRANSITIONS, dutyB recovers ~3/4 of the 4.1% probe
# cost. If it scales with DUTY, dutyA == dutyB and batching buys nothing.
# Decisive, two boots, no code change -- both knobs already exist.
#
# Cell: llama R8 w512. Chosen because the policy correctly PARKS there
# (live f 0.674 < break-even R 0.733), so every probe is a foray into a
# losing configuration and the transition cost is not confounded with a
# steady-state gain. R4 is included as the control: there arming WINS
# (probes were measured +2.1%), so batching must not degrade it.
set -uo pipefail
cd /data/smcho/self-spec-moe

GPU="${E95_GPU:-0}"
PHASE=research/95_c3_deploy
LOG="$PHASE/logs"; mkdir -p "$LOG" "$PHASE/data"

export PATH="/data/smcho/self-spec-moe/.venv/bin:$PATH"
export HF_HOME=/data/smcho/huggingface
export VLLM_CACHE_ROOT=/data/smcho/.cache/vllm
export TRITON_CACHE_DIR=/data/smcho/.cache/triton
export TMPDIR=/data/smcho/tmp
export CUDA_VISIBLE_DEVICES="$GPU"

export VLLM_SELF_SPEC_DRAFT_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1
export VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1
export VLLM_SELF_SPEC_SHARED_KV=1 VLLM_SELF_SPEC_DRAFT_KV_SINKS=16
export VLLM_SELF_SPEC_CPU_ORCH=1 VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE=1
export VLLM_SELF_SPEC_SKIP_PREFILL_DRAFT=1
export VLLM_SELF_SPEC_DRAFT_FULLCG=1 VLLM_SELF_SPEC_DRAFT_STEP0_FULL_CG=1
export VLLM_SELF_SPEC_DRAFT_WHOLECHAIN=1 VLLM_ALLOW_INSECURE_SERIALIZATION=1

cleanup() {
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "${GPU%%,*}" 2>/dev/null); do
    kill -9 "$p" 2>/dev/null
  done
  sleep 5
}

run_arm() {  # name burst interval
  local name="$1" burst="$2" interval="$3"
  local out="$PHASE/data/e1p_${name}.json"
  if [ -f "$out" ]; then echo "[E1p] skip $name (exists)"; return; fi
  echo "[E1p] === $name burst=$burst interval=$interval ==="
  cleanup
  env VLLM_SELF_SPEC_ACCEPT_PROBE_BURST="$burst" \
      VLLM_SELF_SPEC_ACCEPT_PROBE_INTERVAL="$interval" \
      E95_ARCH=llama E95_WINDOW=512 E95_SEED=0 E95_REGIMES=R8,R4 E95_ITERS=3 \
      E95_OUT="$out" \
      timeout 2400 .venv/bin/python "$PHASE/scripts/run_e0_envelope.py" \
        >> "$LOG/e1p_probe.log" 2>&1
  [ $? -ne 0 ] && echo "[E1p] FAILED $name"
}

run_arm dutyA 2 64
run_arm dutyB 8 256
cleanup
echo "[E1p] done"

.venv/bin/python - <<'PY'
import json, pathlib
D = pathlib.Path("research/95_c3_deploy/data")
ar = json.loads((D / "e0_llama_woff_s0.json").read_text())["regimes"]
parked = json.loads((D / "diag_off_llama.json").read_text())["regimes"]
print(f"\n{'arm':8s} {'R8 tok/s':>9s} {'S(R8)':>7s} {'vs parked':>10s}   "
      f"{'R4 tok/s':>9s} {'S(R4)':>7s}")
print(f"{'parked':8s} {parked['R8']['toks']:9.1f} "
      f"{parked['R8']['toks']/ar['R8']['toks']:7.4f} {'--':>10s}   "
      f"{parked['R4']['toks']:9.1f} {parked['R4']['toks']/ar['R4']['toks']:7.4f}")
for name in ("dutyA", "dutyB"):
    f = D / f"e1p_{name}.json"
    if not f.exists():
        continue
    r = json.loads(f.read_text())["regimes"]
    d = (r["R8"]["toks"] / parked["R8"]["toks"] - 1) * 100
    print(f"{name:8s} {r['R8']['toks']:9.1f} "
          f"{r['R8']['toks']/ar['R8']['toks']:7.4f} {d:+9.2f}%   "
          f"{r['R4']['toks']:9.1f} {r['R4']['toks']/ar['R4']['toks']:7.4f}")
print("\ndutyB ~= parked  -> cost is TRANSITIONS (batch the exploration)")
print("dutyB ~= dutyA   -> cost is DUTY (batching buys nothing; need a "
      "parked-mode probe)")
print("R4 must not regress in either arm -- probes WIN there (+2.1%).")
PY
