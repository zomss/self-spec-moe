#!/usr/bin/env bash
# Phase 45 A/B driver: baseline vs CPU-orch reductions, on one model.
# Runs, serially with own teardown, the fine CPU profile + the accept
# (losslessness) check for BOTH the baseline binary path and the
# VLLM_SELF_SPEC_CPU_ORCH path, so before/after CPU breakdown + identical
# accept_len/token-ids can be compared.
#
# Usage:  run_ab.sh <model_tag>   with env CP_MODEL / CP_BATCH / CP_K / CP_DP set.
#   model_tag: filename tag, e.g. qwen15moe or qwen30b.
set -u
cd /data/smcho/ssm-cpu
export PYTHONPATH=/data/smcho/ssm-cpu
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
export NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=1
export HF_HUB_OFFLINE=1
PY=/data/smcho/self-spec-moe/.venv/bin/python
SCRIPT=research/45_cpu_orch/scripts/cpu_profile.py
LOGDIR=research/45_cpu_orch/logs
TAG="${1:-qwen15moe}"

run() {  # $1=job $2=tag-suffix $3=knobs $4=logname
  echo "=== [$4] job=$1 knobs=[$3] ==="
  CP_JOB="$1" CP_TAG="${TAG}_$2" CP_KNOBS="$3" \
    timeout 1600 "$PY" "$SCRIPT" > "$LOGDIR/$4.log" 2>&1
  echo "exit=$? ($4)"
  grep -E "CP\]" "$LOGDIR/$4.log" | tail -30
  # teardown safety: kill any lingering workers from this run.
  pkill -f "cpu_profile.py" 2>/dev/null; sleep 3
}

run profile base   ""                          "${TAG}_prof_base"
run profile orch   "VLLM_SELF_SPEC_CPU_ORCH=1" "${TAG}_prof_orch"
run accept  base   ""                          "${TAG}_acc_base"
run accept  orch   "VLLM_SELF_SPEC_CPU_ORCH=1" "${TAG}_acc_orch"

echo "=== losslessness diff ==="
"$PY" - "$TAG" <<'PYEOF'
import json, sys, glob, os
tag = sys.argv[1]
d = "research/45_cpu_orch/data"
def load(sfx):
    p = os.path.join(d, f"cp_{tag}_{sfx}_b*_K*_accept*.json")
    fs = glob.glob(p)
    return json.load(open(fs[0])) if fs else None
b = load("acc_base"); o = load("acc_orch")
if not b or not o:
    print("MISSING accept json", bool(b), bool(o)); sys.exit(0)
tb, to = b.get("token_ids"), o.get("token_ids")
al_b = b["run_result"].get("accept_len"); al_o = o["run_result"].get("accept_len")
same = tb == to
print(f"accept_len base={al_b}  orch={al_o}")
print(f"token_ids identical: {same}  (nreqs base={len(tb or [])} orch={len(to or [])})")
if not same and tb and to:
    for i,(x,y) in enumerate(zip(tb,to)):
        if x!=y:
            print(f"  first diff req {i}: base[:8]={x[:8]} orch[:8]={y[:8]}")
            break
PYEOF
