# Phase 61 1-RAIL env for the 236B runs (sourced on BOTH nodes).
# Copy of research/52_two_node_e2e/scripts/env_2node.sh (Phase-47 full spec
# stack) with ONE fabric change: NCCL_IB_HCA restricted to EXACTLY mlx5_0
# (1 NIC per node -> real high-f regime at 236B payload sizes).
REPO=/h/v-sukmincho/self-spec-moe
export PYTHONPATH="$REPO"
export PATH="$REPO/.venv/bin:$PATH"
export HF_HUB_OFFLINE=1
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0

# 1-RAIL fabric: exact-match one HCA. gloo/bootstrap/DP-RPC stay on rail 0
# (enmlx0), NOT the frontend (Phase 52: TCP-RTO tail spikes on the frontend).
export NCCL_IB_HCA='=mlx5_0'
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=enmlx0
export GLOO_SOCKET_IFNAME=enmlx0

# Phase-47 fully-optimized spec stack (draft replica/local-route flags are set
# inside the worker for spec mode).
export VLLM_SELF_SPEC_DRAFT_FULL_CG=1
export VLLM_SELF_SPEC_COMPILE_CONSISTENT=1
export VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1
export VLLM_SELF_SPEC_LOG_A2A_COUNTS=1

export W7_MODEL=Qwen/Qwen3-30B-A3B
export W7_TP=1 W7_TRC=0 W7_EAGER=0
export W7_DRAFT_QUANT=fp8
export W7_OUTLEN=160 W7_SHORTLEN=32 W7_ITERS=3 W7_WARMUP=2
export W7_GPU_MEM=0.90
export W7_KS=2
export W7_BATCHES=32,64,128
export W7_OUT="$REPO/research/61_236b_one_rail/data"
export W7_TAG=dsv2_1rail
export W7_MASTER_IP=192.168.0.17
export W7_MASTER_PORT=13355
export W7_NODES=2 W7_LOCAL_WORLD=8
