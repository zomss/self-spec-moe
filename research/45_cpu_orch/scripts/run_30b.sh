#!/usr/bin/env bash
# Phase 45: confirm on Qwen3-30B-A3B DP8 K2 b64 forced-PCIe (idle GPUs, full mem).
set -u
cd /data/smcho/ssm-cpu
export PYTHONPATH=/data/smcho/ssm-cpu VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
export NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=1 HF_HUB_OFFLINE=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/data/smcho/self-spec-moe/.venv/bin/python
L=research/45_cpu_orch/logs
D=research/45_cpu_orch/data
export CP_MODEL="Qwen/Qwen3-30B-A3B" CS_MODEL="Qwen/Qwen3-30B-A3B"
export CP_GPU_MEM=0.90 CS_GPU_MEM=0.90 CP_FINE=0
td(){ pkill -f cpu_profile.py 2>/dev/null; pkill -f cycle_speedup.py 2>/dev/null; sleep 5; }

echo "=== 30B fine profile (base) ==="
CP_JOB=profile CP_TAG=qwen30b_base CP_FINE=1 CP_KNOBS="" timeout 2400 "$PY" research/45_cpu_orch/scripts/cpu_profile.py > "$L/qwen30b_prof_base.log" 2>&1
echo exit=$?; grep -E "CP\]" "$L/qwen30b_prof_base.log" | tail -25; td

echo "=== 30B base accept ==="
CP_JOB=accept CP_TAG=qwen30b_accbase CP_KNOBS="" timeout 2400 "$PY" research/45_cpu_orch/scripts/cpu_profile.py > "$L/qwen30b_accbase.log" 2>&1
echo exit=$?; grep -E "CP\] accept_len|CP\] wrote" "$L/qwen30b_accbase.log" | tail; td

echo "=== 30B orch accept ==="
CP_JOB=accept CP_TAG=qwen30b_accorch CP_KNOBS="VLLM_SELF_SPEC_CPU_ORCH=1" timeout 2400 "$PY" research/45_cpu_orch/scripts/cpu_profile.py > "$L/qwen30b_accorch.log" 2>&1
echo exit=$?; grep -E "CP\] accept_len|CP\] wrote" "$L/qwen30b_accorch.log" | tail; td

echo "=== 30B losslessness diff ==="
OJ=$(ls "$D"/cp_qwen30b_accorch_b64_K2_accept*.json 2>/dev/null | head -1)
"$PY" research/45_cpu_orch/scripts/diff_accept.py "$D/cp_qwen30b_accbase_b64_K2_accept.json" "$OJ" 2>&1

echo "=== 30B cycle + speedup ==="
CS_TAG=qwen30b timeout 3000 "$PY" research/45_cpu_orch/scripts/cycle_speedup.py > "$L/qwen30b_cyclespeedup.log" 2>&1
echo exit=$?; grep -E "CS\]" "$L/qwen30b_cyclespeedup.log" | tail -12; td
echo "=== 30B DONE ==="
