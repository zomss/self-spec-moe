#!/usr/bin/env bash
# W8d: is the profiled boot's S_ss depressed by the profiler's own syncs?
# Same fixed-256 protocol, uncond, b32, NO profiling. If unperturbed
# S_ss ~= profiled S_ss, the residual is real serving work; if it is
# much higher, the "uncovered" 13-22% is instrument overhead.
set -uo pipefail
cd /data/smcho/self-spec-moe
PHASE=research/96_selector_foundations
export PATH="/data/smcho/self-spec-moe/.venv/bin:$PATH"
export HF_HOME=/data/smcho/huggingface VLLM_CACHE_ROOT=/data/smcho/.cache/vllm
export TRITON_CACHE_DIR=/data/smcho/.cache/triton TMPDIR=/data/smcho/tmp
COMMON="VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1 VLLM_SELF_SPEC_CPU_ORCH=1"
SHARED="$COMMON VLLM_SELF_SPEC_SHARED_KV=1 VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE=1 VLLM_SELF_SPEC_SKIP_PREFILL_DRAFT=1 VLLM_SELF_SPEC_DRAFT_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1"
cleanup() { for g in ${1//,/ }; do for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$g" 2>/dev/null); do kill -9 "$p" 2>/dev/null; done; done
  w=0; while [ $w -lt 180 ]; do busy=0; for g in ${1//,/ }; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null); [ "${u:-0}" -gt 2000 ] && busy=1; done; [ $busy -eq 0 ] && break; sleep 10; w=$((w+10)); done; sleep 5; }

cleanup 0
env CUDA_VISIBLE_DEVICES=0 G93_MODEL=deepseek-ai/DeepSeek-V2-Lite G93_TP=1 \
    G93_DRAFT="/data/smcho/ckpts/DeepSeek-V2-Lite-W8A16-INT8-chan" G93_K=4 \
    G93_BATCHES=32 G93_DATASETS=R2 G93_ITERS=3 G93_CEILING=2048 G93_MAXLEN=24576 \
    G93_TUNE=0 G93_FIXED_LEN=256 G93_TAG=w8d_mla_unc_noprof \
    G93_OUT="$PHASE/data/w8/w8d_mla_uncond_noprof_b32.json" $SHARED \
    timeout 1800 .venv/bin/python research/93_c1_grid/scripts/run_grid.py \
    > "$PHASE/logs/w8d_mla.log" 2>&1 || echo "FAILED mla"
cleanup 0,1
env CUDA_VISIBLE_DEVICES=0,1 G93_MODEL=Qwen/Qwen3-30B-A3B G93_TP=2 \
    G93_DRAFT="/data/smcho/ckpts/Qwen3-30B-A3B-W4A16-INT4-sym" G93_K=4 \
    G93_BATCHES=32 G93_DATASETS=R2 G93_ITERS=3 G93_CEILING=16384 G93_MAXLEN=24576 \
    G93_TUNE=0 G93_FIXED_LEN=256 G93_TAG=w8d_moe_unc_noprof \
    G93_OUT="$PHASE/data/w8/w8d_moe_uncond_noprof_b32.json" $SHARED \
    timeout 3600 .venv/bin/python research/93_c1_grid/scripts/run_grid.py \
    > "$PHASE/logs/w8d_moe.log" 2>&1 || echo "FAILED moe"
cleanup 0,1
echo "[W8d] done"
