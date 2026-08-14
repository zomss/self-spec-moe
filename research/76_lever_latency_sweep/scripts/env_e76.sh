# Phase 76 env -- standalone-engine lever sweep (NO self-spec harness in the loop).
# Portable: REPO derived from this file's path.
#
# E0/sweep primitive is `vllm bench latency` on a standalone engine (E1/E4 method),
# so none of the W7_* harness vars are set here. What matters is measurement hygiene
# and that every arm sees the SAME env except the intended delta.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH="$REPO"
export PATH="$REPO/.venv/bin:$PATH"     # MLA JIT needs ninja (1.13.0 in .venv/bin)
PY="$REPO/.venv/bin/python"

export HF_HUB_OFFLINE=1
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0

# Root FS on this box fills up unpredictably (other users); ALL our JIT/temp
# artifacts go to /data. Two E0 casualties before this was learned:
#  - FlashInfer fused_moe ninja build: "No space left on device" (nvcc tmp)
#  - fp8_per_block first forward: DeepGEMM runtime cubin compiled EMPTY into
#    ~/.tensorrt_llm/cache -> "Assertion failed: !cubin.empty()" (poisoned
#    cache purged; TRTLLM_DG_CACHE_DIR moves it off root for good)
export TMPDIR=/data/smcho/tmp
export TRTLLM_DG_CACHE_DIR=/data/smcho/cache/trtllm_dg
mkdir -p "$TMPDIR" "$TRTLLM_DG_CACHE_DIR"
# DeepGEMM's runtime JIT popen()s bare `nvcc` when CUDA_HOME is unset -> the
# EngineCore workers see "sh: nvcc: not found", compile silently produces no
# cubin (stderr is not captured), then "Assertion failed: !cubin.empty()".
export CUDA_HOME=/usr/local/cuda
export TRTLLM_DG_NVCC_COMPILER=/usr/local/cuda/bin/nvcc
export PATH="/usr/local/cuda/bin:$PATH"
# User moved these to /data (~/.bashrc, 2026-07-10); repeat here so the runner
# works from any shell. ~/.cache/huggingface no longer exists.
export HF_HOME=/data/smcho/huggingface
export VLLM_CACHE_ROOT=/data/smcho/.cache/vllm
export TRITON_CACHE_DIR=/data/smcho/.cache/triton

# Phase-74 measurement hygiene (non-negotiable).
unset VLLM_SELF_SPEC_COMPILE_CONSISTENT
unset VLLM_SELF_SPEC_PROFILE
# Lever knobs must be OFF unless an arm sets them explicitly.
unset VLLM_SELF_SPEC_SKIP_A2A VLLM_SELF_SPEC_LOCAL_ROUTE VLLM_DISABLED_KERNELS

# GPU assignment: dense arms on one GPU, MoE/MLA arms on four (DP4+EP4).
# This box's GPUs 2-5 are reserved for other work -- phase 76 uses 0,1,6,7 only.
export E76_GPU="${E76_GPU:-0}"
export E76_MOE_GPUS="${E76_MOE_GPUS:-0,1,6,7}"

# Models + P75 checkpoints (dense W4 reused; skip-if-missing like E1).
export E76_DENSE="Qwen/Qwen2.5-7B-Instruct"           # 28 layers
export E76_MOE="Qwen/Qwen3-30B-A3B"                   # 48 layers, GQA
export E76_MLA="deepseek-ai/DeepSeek-V2-Lite"         # 27 layers, MLA
CKPT_ROOT="${CKPT_ROOT:-/data/smcho/ckpts}"
export E76_W4="$CKPT_ROOT/Qwen2.5-7B-Instruct-W4A16-INT4-sym"

# Force the Marlin mixed-precision kernel (else vLLM auto-picks Machete on SM90).
e76_force_marlin(){
  export VLLM_DISABLED_KERNELS="MacheteLinearKernel,CutlassW4A8LinearKernel,AllSparkLinearKernel"
}
e76_auto_kernel(){ unset VLLM_DISABLED_KERNELS; }
