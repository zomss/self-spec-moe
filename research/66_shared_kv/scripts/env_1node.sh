# Phase 62 single-node env (h107, DP8/EP8): Phase-52 self-spec stack on ONE
# node. Derived from research/52_two_node_e2e/scripts/env_2node.sh with
# W7_NODES=1 W7_LOCAL_WORLD=8; the draft quant/replica/local-route and the
# window-KV knobs are set PER ARM by run_arm.sh (not here).
REPO=/h/v-sukmincho/self-spec-moe
export PYTHONPATH="$REPO"
export PATH="$REPO/.venv/bin:$PATH"
export HF_HUB_OFFLINE=1
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0

# Same NCCL/gloo rails as the proven 1-node runs (run_1node_quick.sh sourced
# env_2node.sh unchanged); on one node these only pin the bootstrap ifaces.
export NCCL_IB_HCA='^mlx5_8'
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=enmlx0
export GLOO_SOCKET_IFNAME=enmlx0

# Phase-47/52 optimized self-spec stack. CHAIN_PIECEWISE keeps the draft
# chain's attention EAGER with a per-step metadata rebuild -- required for
# the Phase-62 window-KV draft (VLLM_SELF_SPEC_DRAFT_KV_WINDOW) to engage.
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

# Fail fast on a wedged collective so the retry runner recovers in ~2 min.
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=120
