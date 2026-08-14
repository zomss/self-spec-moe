#!/bin/bash
# E6d: deep-cell + window-axis + concurrency co-optimization.
cd /data/smcho/self-spec-moe
source research/76_lever_latency_sweep/scripts/env_e76.sh
unset VLLM_SELF_SPEC_COMPILE_CONSISTENT VLLM_SELF_SPEC_PROFILE
export VLLM_SELF_SPEC_DRAFT_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1
export VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1
export VLLM_SELF_SPEC_SHARED_KV=1 VLLM_SELF_SPEC_DRAFT_KV_SINKS=16
export VLLM_SELF_SPEC_CPU_ORCH=1 VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE=1
export VLLM_SELF_SPEC_SKIP_PREFILL_DRAFT=1
export VLLM_SELF_SPEC_DRAFT_FULLCG=1 VLLM_SELF_SPEC_DRAFT_STEP0_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_WHOLECHAIN=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_DISABLED_KERNELS=MacheteLinearKernel,CutlassW4A8LinearKernel,AllSparkLinearKernel
export CUDA_VISIBLE_DEVICES=0
CPS=research/82_runtime_switching/scripts/compile_policy.py
for W in 512 4096; do
  for arm in off k2 k3; do
    echo "[e6d-compile] win$W arm=$arm ($(date +%H:%M:%S))"
    env VLLM_SELF_SPEC_DRAFT_KV_WINDOW=$W COMPILE_MODEL=Qwen/Qwen3-8B \
      COMPILE_DRAFT=/data/smcho/ckpts/Qwen3-8B-W4A8-gptq \
      COMPILE_BATCHES=32,48 COMPILE_CTXS=2000,6000 \
      COMPILE_CELLS=cells_e6d_w$W.csv COMPILE_TABLE=table_e6d_w$W.json \
      timeout -k 30 1500 .venv/bin/python $CPS --measure $arm \
      || echo "[e6d-compile] win$W $arm FAILED"
  done
  env COMPILE_BATCHES=32,48 COMPILE_CTXS=2000,6000 \
    COMPILE_CELLS=cells_e6d_w$W.csv COMPILE_TABLE=table_e6d_w$W.json \
    .venv/bin/python $CPS --solve
done
.venv/bin/python - <<'PYEOF'
import json
base = json.load(open("research/91_bandit_stage3/data/policy_table_e6v3.json"))
for W in (512, 4096):
    t = json.load(open(f"research/82_runtime_switching/data/table_e6d_w{W}.json"))
    cells = [c for c in base["cells"]] + t["cells"]
    json.dump({"cells": cells},
              open(f"research/91_bandit_stage3/data/policy_table_e6d_w{W}.json", "w"), indent=1)
print("[e6d] tables merged")
PYEOF
export E6_PROMPTS=16 E6_GROUP=8 E6_MAXTOK=8192 E6_THINK=1
run_arm () {  # tag window maxseqs
  echo "=== E6d $1 ($(date +%H:%M:%S)) ==="
  env VLLM_SELF_SPEC_DRAFT_KV_WINDOW=$2 \
    VLLM_SELF_SPEC_POLICY_FILE=research/91_bandit_stage3/data/policy_table_e6d_w$2.json \
    E6_ARM=policy E6_K=3 E6_MAXSEQS=$3 \
    timeout -k 30 3600 .venv/bin/python research/91_bandit_stage3/scripts/e6_rollout.py
  mv research/91_bandit_stage3/data/e6_policy.json \
     research/91_bandit_stage3/data/e6d_$1.json 2>/dev/null
}
run_arm w512_b64 512 64
run_arm w4096_b64 4096 64
run_arm w512_waves48 512 48
echo "=== E6d DONE ($(date +%H:%M:%S)) ==="
