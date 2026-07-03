# Shared env for Phase 49 2-node runs (sourced on BOTH nodes).
REPO=/h/v-sukmincho/self-spec-moe
export PYTHONPATH="$REPO"
export PATH="$REPO/.venv/bin:$PATH"
export HF_HUB_OFFLINE=1
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0

# REAL fabric: NO forced-PCIe knobs. RoCEv2 rails for NCCL, frontend for
# bootstrap/gloo/DP-RPC.
export NCCL_IB_HCA='^mlx5_8'
export NCCL_IB_GID_INDEX=3
# gloo/bootstrap/DP-RPC on rail 0 (enmlx0), NOT the frontend: cross-DP TCP
# syncs on the frontend showed ~200ms-quantized tail spikes (TCP RTO) that the
# 16-rank lockstep spec cycle amplifies into every-cycle verify waits.
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
export W7_OUT="$REPO/research/52_two_node_e2e/data"
export W7_TAG=qwen30b_2node
export W7_MASTER_IP=192.168.0.17
export W7_MASTER_PORT=13355
export W7_NODES=2 W7_LOCAL_WORLD=8
