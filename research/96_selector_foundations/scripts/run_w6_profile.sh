#!/usr/bin/env bash
# W6: full 36-layer sensitivity profile + count-{4,8} ladder, self-chaining.
# Phase A: single-layer boots for all 36 layers (L2/8/18/30 exist), w512,
#   R5+R1, SEED 0 ONLY (disclosed deviation: cross-seed tau measured 1.00;
#   the profile only SHORTLISTS -- Round-2 decision boots use both seeds).
#   Two passes (second retries JIT-hang casualties).
# Phase B: compute cheap-tier sets (4 and 8 cheapest by mean R5/R1
#   sensitivity), then FULL-protocol boots (2 seeds) for count4/count8.
set -uo pipefail
cd /data/smcho/self-spec-moe
PHASE=research/96_selector_foundations
LOG="$PHASE/logs"; mkdir -p "$LOG" "$PHASE/data/w6"
export PATH="/data/smcho/self-spec-moe/.venv/bin:$PATH"
export HF_HOME=/data/smcho/huggingface VLLM_CACHE_ROOT=/data/smcho/.cache/vllm
export TRITON_CACHE_DIR=/data/smcho/.cache/triton TMPDIR=/data/smcho/tmp
export VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1
export VLLM_SELF_SPEC_CPU_ORCH=1 VLLM_SELF_SPEC_SHARED_KV=1
export VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE=1 VLLM_SELF_SPEC_SKIP_PREFILL_DRAFT=1
export VLLM_SELF_SPEC_DRAFT_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1
export VLLM_DISABLED_KERNELS=MacheteLinearKernel,CutlassW4A8LinearKernel,AllSparkLinearKernel

cleanup() {
  local gpu="$1"
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$gpu" 2>/dev/null); do
    kill -9 "$p" 2>/dev/null; done
  local w=0
  while [ $w -lt 180 ]; do
    local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" 2>/dev/null)
    [ "${u:-0}" -le 2000 ] && break; sleep 10; w=$((w + 10)); done
  sleep 5
}

boot() {  # gpu skipset seeds outfile
  local gpu="$1" sk="$2" seeds="$3" out="$4"
  [ -f "$out" ] && return 0
  cleanup "$gpu"
  env CUDA_VISIBLE_DEVICES="$gpu" W6_SKIP="$sk" W6_WINDOW=512 \
      W6_REGIMES="R5,R1" W6_SEEDS="$seeds" W6_OUT="$out" \
      timeout 1500 .venv/bin/python "$PHASE/scripts/run_w6_calib.py" \
        >> "$LOG/w6prof_$(basename "$out" .json).log" 2>&1 || true
}

lane() {  # gpu parity
  local gpu="$1" par="$2"
  for pass in 1 2; do
    for L in $(seq 0 35); do
      [ $((L % 2)) -ne "$par" ] && continue
      boot "$gpu" "$L" "0" "$PHASE/data/w6/w6sens_dense_L${L}.json"
    done
  done
  cleanup "$gpu"
}
lane 0 0 & P0=$!
lane 1 1 & P1=$!
wait $P0 $P1
echo "[W6prof] phase A complete: $(ls $PHASE/data/w6/w6sens_dense_L*.json | wc -l)/36"

# Phase B: cheap-tier sets from the profile
SETS=$(.venv/bin/python - <<'EOF'
import json, glob
base = json.load(open("research/96_selector_foundations/data/w6/w6cal_dense_s-none_w512.json"))["cells"]
def f_of(c, rid, s):
    a = [r["accept"] for r in c[f"{rid}_s{s}"] if r["accept"]]
    return (sum(a) / len(a) - 1) / 4
sens = {}
for p in glob.glob("research/96_selector_foundations/data/w6/w6sens_dense_L*.json"):
    L = int(p.split("_L")[1].split(".")[0])
    c = json.load(open(p))["cells"]
    v = []
    for rid in ("R5", "R1"):
        for s in (0, 1):
            if f"{rid}_s{s}" in c:
                v.append(f_of(base, rid, s) - f_of(c, rid, s))
    sens[L] = sum(v) / len(v)
order = sorted(sens, key=sens.get)
print(",".join(map(str, sorted(order[:4]))) + " " + ",".join(map(str, sorted(order[:8]))))
EOF
)
C4=$(echo "$SETS" | cut -d' ' -f1); C8=$(echo "$SETS" | cut -d' ' -f2)
echo "[W6prof] cheap sets: count4={$C4} count8={$C8}"
boot 0 "$C4" "0,1" "$PHASE/data/w6/w6count_dense_c4.json" &
boot 1 "$C8" "0,1" "$PHASE/data/w6/w6count_dense_c8.json" &
wait
cleanup 0; cleanup 1
echo "[W6prof] all complete"
