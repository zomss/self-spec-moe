# Phase 63 8-RAIL env for the 236B 16k-context runs (sourced on BOTH nodes).
# Verbatim clone of research/61_236b_one_rail/scripts/env_8rail_2node.sh
# (which produced the same-HEAD 1k baselines 283.0/866.3/1347.7 tok/s):
# all rails except the frontend mlx5_8; W7_* defaults overridden by the runner.
REPO=/h/v-sukmincho/self-spec-moe
export PYTHONPATH="$REPO"
export PATH="$REPO/.venv/bin:$PATH"
export HF_HUB_OFFLINE=1
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0

# 8-RAIL fabric: exclude only the frontend HCA. gloo/bootstrap/DP-RPC stay on
# rail 0 (enmlx0), NOT the frontend (Phase 52: TCP-RTO tail spikes there).
export NCCL_IB_HCA='^mlx5_8'
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=enmlx0
export GLOO_SOCKET_IFNAME=enmlx0

# Phase-47 fully-optimized spec stack (inert for nospec; kept identical to
# the Phase 61 baseline env).
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
export W7_OUT="$REPO/research/63_mla_context/data"
export W7_TAG=dsv2_8rail_16k
export W7_MASTER_IP=192.168.0.17
export W7_MASTER_PORT=14100
export W7_NODES=2 W7_LOCAL_WORLD=8
