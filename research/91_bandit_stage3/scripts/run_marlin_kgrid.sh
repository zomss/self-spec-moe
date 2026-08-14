#!/bin/bash
# GPU 0: compile the W4A16(Marlin) K-grid table (off/k2/k3/k4), then
# run argmax kmax3 vs kmax4 drift arms -- the K-grid question on the
# wedge-free kernel.
set -e
cd /data/smcho/self-spec-moe
source research/76_lever_latency_sweep/scripts/env_e76.sh
unset VLLM_SELF_SPEC_COMPILE_CONSISTENT VLLM_SELF_SPEC_PROFILE
export VLLM_SELF_SPEC_DRAFT_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1
export VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1
export VLLM_SELF_SPEC_SHARED_KV=1 VLLM_SELF_SPEC_DRAFT_KV_WINDOW=512 VLLM_SELF_SPEC_DRAFT_KV_SINKS=16
export VLLM_SELF_SPEC_CPU_ORCH=1 VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE=1
export VLLM_SELF_SPEC_SKIP_PREFILL_DRAFT=1
export VLLM_SELF_SPEC_DRAFT_FULLCG=1 VLLM_SELF_SPEC_DRAFT_STEP0_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_WHOLECHAIN=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export CUDA_VISIBLE_DEVICES=0
export COMPILE_MODEL=Qwen/Qwen3-8B
export COMPILE_DRAFT=/data/smcho/ckpts/Qwen3-8B-W4A16-INT4
export COMPILE_CELLS=policy_cells_w4grid.csv
export COMPILE_TABLE=policy_table_w4grid.json
for arm in off k2 k3 k4; do
  echo "[marlin-compile] arm=$arm ($(date +%H:%M:%S))"
  .venv/bin/python research/82_runtime_switching/scripts/compile_policy.py --measure $arm
done
.venv/bin/python research/82_runtime_switching/scripts/compile_policy.py --solve
cp research/82_runtime_switching/data/policy_table_w4grid.json research/91_bandit_stage3/data/
.venv/bin/python - <<'PYEOF'
import json
t = json.load(open("research/91_bandit_stage3/data/policy_table_w4grid.json"))
k3 = {"cells": [dict(c, options=[o for o in c["options"] if o["K"] <= 3]) for c in t["cells"]]}
json.dump(k3, open("research/91_bandit_stage3/data/policy_table_w4grid_k3.json", "w"), indent=1)
print("k3-filtered table written")
PYEOF
export E3_MODE=run E3_EPS=0,0.1,0.2,0.3,0.4 E3_GATE=2.45 E3_POLL=10
export E3_MAXTOK=3072 E3_EOS=1
export E3_DRAFT=/data/smcho/ckpts/Qwen3-8B-W4A16-INT4
run_arm () {
  local name=$1 k=$2 table=$3
  echo "=== MK $name ($(date +%H:%M:%S)) ==="
  env VLLM_SELF_SPEC_POLICY_FILE=$table E3_ARM=$name E3_SPEC=1 E3_K=$k E3_BG_DETECT=1 \
    timeout -k 30 1800 .venv/bin/python research/89_dram_lever_swap/scripts/e3_rl_demo.py \
    || echo "=== MK $name FAILED rc=$? ==="
}
run_arm marlin_k3 3 research/91_bandit_stage3/data/policy_table_w4grid_k3.json
run_arm marlin_k4 4 research/91_bandit_stage3/data/policy_table_w4grid.json
echo "=== MK DONE ($(date +%H:%M:%S)) ==="
