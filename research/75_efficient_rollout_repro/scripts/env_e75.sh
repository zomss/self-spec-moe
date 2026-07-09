# Phase 75 env -- single GPU, DENSE model (Qwen2.5-7B-Instruct), TP=1, DP=1, no EP.
# Portable: REPO derived from this file's path. GPU via ${E75_GPU:-0}.
#
# This is EfficientRollout's setting, not ours: one dense model, one GPU, batch 1.
# Deliberately NOT sourcing 74's env_local.sh (that one is MoE/EP4/DP4).
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH="$REPO"
export PATH="$REPO/.venv/bin:$PATH"

export CUDA_VISIBLE_DEVICES="${E75_GPU:-0}"
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
unset NCCL_IB_HCA NCCL_IB_GID_INDEX

export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0

# Phase-74 measurement hygiene (non-negotiable; these two taxed the spec path ~2x
# and ~25% respectively and produced the bogus "self-spec loses 2x" headline).
unset VLLM_SELF_SPEC_COMPILE_CONSISTENT
unset VLLM_SELF_SPEC_PROFILE

# Self-spec stack for the draft_model proposer.
export VLLM_SELF_SPEC_DRAFT_FULL_CG=1
export VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1
export VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1
export VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1
# SHARED_KV is a CONTROLLED KNOB in E2, not a default. The paper's drafter computes
# its own KV; SHARED_KV=1 would make our quantized draft attend the target's exact KV
# and inflate tau. Runners set it explicitly.
export VLLM_SELF_SPEC_SHARED_KV="${E75_SHARED_KV:-0}"

# Dense model, single GPU.
export W7_MODEL="${E75_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
export W7_TP=1 W7_TRC=0 W7_EAGER=0
export W7_NODES=1 W7_LOCAL_WORLD=1
export W7_EP=0                      # dense: expert-parallel is meaningless
export W7_GPU_MEM=0.85
export W7_MASTER_IP=127.0.0.1
export W7_SPEC_METHOD=draft_model

# Their point: batch 1, seq 2k.
export W7_CTX_TOKENS="${E75_CTX:-2048}"
export W7_MAX_MODEL_LEN="${E75_MAXLEN:-4096}"
export W7_MAX_NUM_BATCHED=8192
export W7_OUTLEN=160 W7_SHORTLEN=32

export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=180

# Force the PAPER's kernel (Marlin) by disabling everything ahead of it in vLLM's
# CUDA mixed-precision priority list. Unset -> vLLM auto-picks Machete.
e75_force_marlin(){
  export VLLM_DISABLED_KERNELS="MacheteLinearKernel,CutlassW4A8LinearKernel,AllSparkLinearKernel"
}
e75_auto_kernel(){ unset VLLM_DISABLED_KERNELS; }
