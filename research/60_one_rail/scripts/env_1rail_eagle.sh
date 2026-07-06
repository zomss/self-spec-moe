# Phase 60 1-RAIL env for the nospec baseline + EAGLE arms (sourced on BOTH
# nodes). Copy of research/57_large_ep_spec_strategy/scripts/env_eagle_2node.sh
# with ONE change to the fabric: NCCL_IB_HCA restricted to EXACTLY mlx5_0
# (1 NIC per node instead of 8 rails) -> per-GPU inter-node bandwidth /8,
# the common production topology, pushing comm fraction f to the HIGH regime.
REPO=/h/v-sukmincho/self-spec-moe
export PYTHONPATH="$REPO"
export PATH="$REPO/.venv/bin:$PATH"
export HF_HUB_OFFLINE=1
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0

# 1-RAIL fabric: exact-match ONLY mlx5_0 (8-rail env excluded just the
# frontend: '^mlx5_8'). Keep RoCEv2 GID and frontend ifname for gloo/bootstrap.
export NCCL_IB_HCA='=mlx5_0'
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=enmlx0
export GLOO_SOCKET_IFNAME=enmlx0

# NO VLLM_SELF_SPEC_* flags here: EAGLE is a separate drafter model, not the
# self-spec draft path.

export W7_MODEL=Qwen/Qwen3-30B-A3B
export W7_TP=1 W7_TRC=0 W7_EAGER=0
export W7_OUTLEN=160 W7_SHORTLEN=32 W7_ITERS=3 W7_WARMUP=2
export W7_GPU_MEM=0.90
export W7_OUT="$REPO/research/60_one_rail/data"
export W7_TAG=q30b_1rail
export W7_MASTER_IP=192.168.0.17
export W7_MASTER_PORT=13700
export W7_NODES=2 W7_LOCAL_WORLD=8

# EAGLE3 drafter selection (real trained head for Qwen3-30B-A3B).
export W7_SPEC_METHOD=eagle3
export W7_SPEC_MODEL=Tengyunw/qwen3_30b_moe_eagle3

# Same DP16-EAGLE mitigations as Phase 57: async sched off, fail-fast timeout
# (1-rail comm is ~8x slower; give the collective more headroom than 120s).
export W7_ASYNC_SCHED=0
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=240
