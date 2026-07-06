# Phase 60 1-RAIL env for the World A self-spec arm (sourced on BOTH nodes).
# Copy of research/52_two_node_e2e/scripts/env_2node.sh (Phase-47 full spec
# stack) with ONE fabric change: NCCL_IB_HCA restricted to EXACTLY mlx5_0
# (1 NIC per node -> real high-f regime).
REPO=/h/v-sukmincho/self-spec-moe
export PYTHONPATH="$REPO"
export PATH="$REPO/.venv/bin:$PATH"
export HF_HUB_OFFLINE=1
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0

# 1-RAIL fabric (see env_1rail_eagle.sh).
export NCCL_IB_HCA='=mlx5_0'
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=enmlx0
export GLOO_SOCKET_IFNAME=enmlx0

# Phase-47 fully-optimized spec stack (draft replica/local-route flags are set
# inside the worker for spec mode; worker defaults W7_DRAFT_LOCAL_ROUTE=1,
# W7_DRAFT_FULL_REPLICA=1 -> FP8 full-replica comm-free draft).
export VLLM_SELF_SPEC_DRAFT_FULL_CG=1
export VLLM_SELF_SPEC_COMPILE_CONSISTENT=1
export VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1
export VLLM_SELF_SPEC_LOG_A2A_COUNTS=1
# Explicit default (0 = off): no amortized DP coord; leave SKIP_DP_COORD,
# NODE_LOCAL, STEP0_FULL_CG at their defaults (unset).
export VLLM_SELF_SPEC_DRAFT_AMORTIZE_DP_COORD=0

export W7_MODEL=Qwen/Qwen3-30B-A3B
export W7_TP=1 W7_TRC=0 W7_EAGER=0
export W7_DRAFT_QUANT=fp8
export W7_OUTLEN=160 W7_SHORTLEN=32 W7_ITERS=3 W7_WARMUP=2
export W7_GPU_MEM=0.90
export W7_KS=2
export W7_BATCHES=8,32,64
export W7_OUT="$REPO/research/60_one_rail/data"
export W7_TAG=q30b_1rail_worldA
export W7_MASTER_IP=192.168.0.17
export W7_MASTER_PORT=13800
export W7_NODES=2 W7_LOCAL_WORLD=8
