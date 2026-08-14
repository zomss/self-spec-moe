#!/bin/bash
# E6: compile K8 cells -> build the E6 table (K2,K3,K6,K8 -- width-5
# avoided) -> arms: off, policy(fresh), rl_refresh, rl_stale. GPU 0.
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
export VLLM_DISABLED_KERNELS=MacheteLinearKernel,CutlassW4A8LinearKernel,AllSparkLinearKernel
export CUDA_VISIBLE_DEVICES=0

if [ ! -f research/91_bandit_stage3/data/policy_table_e6.json ]; then
  echo "[e6-compile] k8 cells ($(date +%H:%M:%S))"
  env COMPILE_MODEL=Qwen/Qwen3-8B COMPILE_DRAFT=/data/smcho/ckpts/Qwen3-8B-W4A8-gptq \
    COMPILE_CELLS=policy_cells_hum.csv COMPILE_TABLE=policy_table_hum_k8.json \
    timeout -k 30 1200 .venv/bin/python research/82_runtime_switching/scripts/compile_policy.py --measure k8
  env COMPILE_MODEL=Qwen/Qwen3-8B COMPILE_DRAFT=/data/smcho/ckpts/Qwen3-8B-W4A8-gptq \
    COMPILE_CELLS=policy_cells_hum.csv COMPILE_TABLE=policy_table_hum_k8.json \
    .venv/bin/python research/82_runtime_switching/scripts/compile_policy.py --solve
  .venv/bin/python - <<'PYEOF'
import json
t = json.load(open("research/82_runtime_switching/data/policy_table_hum_k8.json"))
keep = {2, 3, 6, 8}   # avoid width-5 (K4); K8 for the deep tail
t = {"cells": [dict(c, options=[o for o in c["options"] if o["K"] in keep])
               for c in t["cells"]]}
json.dump(t, open("research/91_bandit_stage3/data/policy_table_e6.json", "w"), indent=1)
print("[e6-compile] table:", sum(len(c["options"]) for c in t["cells"]), "options")
PYEOF
fi

export E6_PROMPTS=16 E6_GROUP=8 E6_MAXTOK=8192
POL=research/91_bandit_stage3/data/policy_table_e6.json
run_arm () {  # arm extra...
  local arm=$1; shift
  for try in 1 2; do
    echo "=== E6 $arm try$try ($(date +%H:%M:%S)) ==="
    env "$@" E6_ARM=$arm \
      timeout -k 30 3600 .venv/bin/python research/91_bandit_stage3/scripts/e6_rollout.py && return 0
    pgrep -f "scripts/e6_rollout" | while read p; do kill -9 $p; done; sleep 8
  done
  echo "=== E6 $arm GAVE UP ==="
}
run_arm off
run_arm policy VLLM_SELF_SPEC_POLICY_FILE=$POL E6_K=8
run_arm rl_refresh VLLM_SELF_SPEC_POLICY_FILE=$POL E6_K=8
run_arm rl_stale VLLM_SELF_SPEC_POLICY_FILE=$POL E6_K=8
echo "=== E6 DONE ($(date +%H:%M:%S)) ==="
