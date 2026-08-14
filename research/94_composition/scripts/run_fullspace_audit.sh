#!/bin/bash
# C2 close-out: the full-space SAMPLED AUDIT. Boots the fixed-seed
# uniform sample (paper/data/c2_fullspace_audit_plan.json, seed 940805)
# so the "regret vs the full deployable space" claim becomes a measured
# bound instead of an extrapolation. Same recipe/protocol as
# run_fullspace_confirms.sh (isolated cache roots); data merges into
# the fs_dense_* record. Usage: FS_GPU=<n> FS_STRIDE=2 FS_OFFSET=<0|1>.
set -uo pipefail
REPO=/data/smcho/self-spec-moe
PHASE=$REPO/research/94_composition
P82DATA=$REPO/research/82_runtime_switching/data
COMPILE=$REPO/research/93_c1_grid/scripts/compile_cells_93.py
export HF_HOME=/data/smcho/huggingface
export PATH="$REPO/.venv/bin:$PATH"
cd "$REPO"
GPU="${FS_GPU:-6}"; MODEL="Qwen/Qwen3-8B"

COMMON="VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1 VLLM_SELF_SPEC_CPU_ORCH=1"
SHARED="$COMMON VLLM_SELF_SPEC_SHARED_KV=1 VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE=1 VLLM_SELF_SPEC_SKIP_PREFILL_DRAFT=1 VLLM_SELF_SPEC_DRAFT_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1"
HUM="VLLM_DISABLED_KERNELS=MacheteLinearKernel,CutlassW4A8LinearKernel,AllSparkLinearKernel"

gpu_cleanup() {
  local u; u=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$GPU")
  for p in $(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | tr -d ',' | awk -v u="$u" '$2==u{print $1}'); do
    [ "$(ps -o user= -p "$p" 2>/dev/null)" = "smcho" ] || continue
    kill "$p" 2>/dev/null; sleep 2; kill -0 "$p" 2>/dev/null && kill -9 "$p" 2>/dev/null
  done
  sleep 3
}

python3 - "$REPO/paper/data/c2_fullspace_audit_plan.json" <<'PY' > /tmp/fs_audit_plan.txt
import json, sys
plan = json.load(open(sys.argv[1]))["boot_plan"]
CK = {"w4a16": ("/data/smcho/ckpts/Qwen3-8B-W4A16-INT4", ""),
      "w4a8cut": ("/data/smcho/ckpts/Qwen3-8B-W4A8-gptq", ""),
      "w4a8hum": ("/data/smcho/ckpts/Qwen3-8B-W4A8-gptq", "HUM"),
      "w8int8": ("/data/smcho/ckpts/Qwen3-8B-W8A16-INT8-sym", ""),
      "w8fp8": ("/data/smcho/ckpts/Qwen3-8B-W8A16-FP8", "VLLM_TEST_FORCE_FP8_MARLIN=1"),
      "fp8dyn": ("/data/smcho/ckpts/Qwen3-8B-FP8-dynamic", "VLLM_SELF_SPEC_DRAFT_EAGER=1")}
SK = {"skipb2": "2,8", "skipb4": "2,4,8,10"}
for name, ks in sorted(plan.items()):
    parts = [] if name == "base" else name.split("+")
    draft, extra = "MODEL", []
    for p in parts:
        if p in CK:
            draft, e = CK[p]
            if e:
                extra.append(e)
        elif p.startswith("win"):
            w = p[3:]
            extra.append(f"VLLM_SELF_SPEC_DRAFT_KV_WINDOW={w} "
                         f"VLLM_SELF_SPEC_DRAFT_KV_SINKS=16 "
                         f"VLLM_SELF_SPEC_DRAFT_FULLCG=1")
        elif p in SK:
            extra.append(f"VLLM_SELF_SPEC_DRAFT_SKIP_LAYERS={SK[p]}")
    print(f"{name}|{draft}|{' '.join(extra)}|{','.join(map(str, ks))}")
PY

run_pass() {
  local IDX=0
  while IFS='|' read -r name draft extra ks; do
    [ -z "$name" ] && continue
    draft="${draft/\$HOME/$HOME}"
    [ "$draft" = "MODEL" ] && draft="$MODEL"
    extra="${extra/HUM/$HUM}"
    csv="fs_dense_${name//+/_}.csv"
    for K in ${ks//,/ }; do
      if [ $(( IDX % ${FS_STRIDE:-1} )) -ne "${FS_OFFSET:-0}" ]; then IDX=$((IDX+1)); continue; fi
      IDX=$((IDX+1))
      karm="k$K"
      if [ -f "$P82DATA/$csv" ] && grep -q "^${karm}," "$P82DATA/$csv"; then
        echo "[AUDIT] skip $name/$karm"; continue
      fi
      echo "[AUDIT] measure $name/$karm"
      env CUDA_VISIBLE_DEVICES=$GPU COMPILE_MODEL="$MODEL" \
          COMPILE_DRAFT="$draft" COMPILE_TP=1 \
          COMPILE_BATCHES="1,8,32" COMPILE_CTXS="2000,8000,14000" \
          COMPILE_KV_LIMIT=260000 COMPILE_CELLS="$csv" \
          COMPILE_TABLE="fs_dense_${name//+/_}.json" \
          VLLM_CACHE_ROOT="/data/smcho/vllm_cache_fs/${name//+/_}_$karm" \
          $SHARED $extra timeout 3600 .venv/bin/python "$COMPILE" --measure "$karm" \
          >> "$PHASE/logs/fullspace_audit.log" 2>&1 \
          || echo "[AUDIT] FAIL $name/$karm"
      gpu_cleanup
    done
  done < /tmp/fs_audit_plan.txt
}

run_pass            # first pass
run_pass            # retry pass: skip-guard makes this a no-op unless a boot failed
cp -f "$P82DATA"/fs_dense_*.csv "$PHASE/data/" 2>/dev/null
echo "[AUDIT] FULLSPACE-AUDIT-DONE gpu=$GPU offset=${FS_OFFSET:-0}"
