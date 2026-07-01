#!/usr/bin/env bash
# Phase 45 finish driver: the validations still missing after the first A/B.
#   1. orch accept (losslessness) — FINE off (fast, avoids NCCL desync).
#   2. base accept (re-run to pair cleanly with orch under identical conditions).
#   3. cycle+speedup (nospec / spec_base / spec_orch), prefill-cancelled.
#   4. losslessness diff (base vs orch token ids).
# Serial, own teardown after each. Usage: run_finish.sh <tag>  (env CP_MODEL etc.)
set -u
cd /data/smcho/ssm-cpu
export PYTHONPATH=/data/smcho/ssm-cpu
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
export NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=1
export HF_HUB_OFFLINE=1
# Guard against transient GPU-contention stalls tripping the DP collective
# watchdog (the first orch-accept attempt died to an NCCL timeout under
# contention). Longer heartbeat timeout; still fails fast on a real hang.
export NCCL_TIMEOUT=1200
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1200
PY=/data/smcho/self-spec-moe/.venv/bin/python
LOGDIR=research/45_cpu_orch/scripts/../logs
DATADIR=research/45_cpu_orch/data
TAG="${1:-qwen15moe}"
export CP_FINE=0   # accept jobs: no profiler syncs

teardown() { pkill -f "cpu_profile.py" 2>/dev/null; pkill -f "cycle_speedup.py" 2>/dev/null; sleep 4; }

echo "=== [1/3] base accept (FINE off) ==="
CP_JOB=accept CP_TAG="${TAG}_accbase" CP_KNOBS="" \
  timeout 1600 "$PY" research/45_cpu_orch/scripts/cpu_profile.py \
  > "$LOGDIR/${TAG}_accbase2.log" 2>&1
echo "exit=$?"; grep -E "CP\] accept_len|CP\] wrote" "$LOGDIR/${TAG}_accbase2.log" | tail -2
teardown

echo "=== [2/3] orch accept (FINE off) ==="
CP_JOB=accept CP_TAG="${TAG}_accorch" CP_KNOBS="VLLM_SELF_SPEC_CPU_ORCH=1" \
  timeout 1600 "$PY" research/45_cpu_orch/scripts/cpu_profile.py \
  > "$LOGDIR/${TAG}_accorch2.log" 2>&1
echo "exit=$?"; grep -E "CP\] accept_len|CP\] wrote" "$LOGDIR/${TAG}_accorch2.log" | tail -2
teardown

echo "=== losslessness diff ==="
BJ="$DATADIR/cp_${TAG}_accbase_b64_K2_accept.json"
OJ="$DATADIR/cp_${TAG}_accorch_b64_K2_accept.json"
"$PY" research/45_cpu_orch/scripts/diff_accept.py "$BJ" "$OJ" 2>&1

echo "=== [3/3] cycle + speedup (nospec/spec_base/spec_orch) ==="
CS_TAG="$TAG" timeout 2400 "$PY" research/45_cpu_orch/scripts/cycle_speedup.py \
  > "$LOGDIR/${TAG}_cyclespeedup.log" 2>&1
echo "exit=$?"; grep -E "CS\]" "$LOGDIR/${TAG}_cyclespeedup.log" | tail -12
teardown
echo "=== finish done ==="
