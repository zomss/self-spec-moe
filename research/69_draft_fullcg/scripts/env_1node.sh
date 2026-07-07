# Phase 67 single-node env (h107, DP8/EP8): identical to Phase 66 env_1node.sh
# (Phase-52 self-spec stack on ONE node). Sourced by the phase-67 runners; the
# per-arm draft quant/replica/route + window + P65 flags are set by run_prof.sh.
REPO=/h/v-sukmincho/self-spec-moe
export PYTHONPATH="$REPO"
export PATH="$REPO/.venv/bin:$PATH"
export HF_HUB_OFFLINE=1
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0

export NCCL_IB_HCA='^mlx5_8'
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=enmlx0
export GLOO_SOCKET_IFNAME=enmlx0

# Phase-47/52 optimized self-spec stack. CHAIN_PIECEWISE keeps the draft
# chain's attention EAGER with a per-step metadata rebuild -- required for
# the window-KV draft (VLLM_SELF_SPEC_DRAFT_KV_WINDOW) to engage.
export VLLM_SELF_SPEC_DRAFT_FULL_CG=1
export VLLM_SELF_SPEC_COMPILE_CONSISTENT=1
export VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1

export W7_MODEL=Qwen/Qwen3-30B-A3B
export W7_TP=1 W7_TRC=0 W7_EAGER=0
export W7_OUTLEN=160 W7_SHORTLEN=32
export W7_GPU_MEM=0.90
export W7_MASTER_IP=192.168.0.17
export W7_MASTER_PORT=13900
export W7_NODES=1 W7_LOCAL_WORLD=8

export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=120
