# Phase 64 self-spec (draft_model) env for 2-node runs (sourced on BOTH nodes).
# Clone of research/52_two_node_e2e/scripts/env_2node.sh (the proven Phase-52
# draft_model spec stack on the real fabric) with these deltas:
#   - W7_DRAFT_QUANT REMOVED (bf16 EP-routed self-draft; runner also exports
#     it empty per arm) -- Phase 64 arms 2-4 share target weights, no replica.
#   - VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=120 (Phase 62 convention): a wedged
#     collective dies in ~2 min so the retry runner recovers inside TRY_TO.
#   - W7_OUT/W7_TAG/W7_MASTER_PORT point at phase 64 (runner overrides anyway).
REPO=/h/v-sukmincho/self-spec-moe
export PYTHONPATH="$REPO"
export PATH="$REPO/.venv/bin:$PATH"
export HF_HUB_OFFLINE=1
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0

# REAL fabric: NO forced-PCIe knobs. RoCEv2 rails for NCCL, frontend for
# bootstrap/gloo/DP-RPC.
export NCCL_IB_HCA='^mlx5_8'
export NCCL_IB_GID_INDEX=3
# gloo/bootstrap/DP-RPC on rail 0 (enmlx0), NOT the frontend (Phase 52:
# TCP-RTO tail spikes there).
export NCCL_SOCKET_IFNAME=enmlx0
export GLOO_SOCKET_IFNAME=enmlx0

# Phase-47/52 fully-optimized self-spec stack. CHAIN_PIECEWISE keeps the draft
# chain's attention EAGER with a per-step metadata rebuild -- required for the
# Phase-62 window-KV draft (VLLM_SELF_SPEC_DRAFT_KV_WINDOW) to engage.
export VLLM_SELF_SPEC_DRAFT_FULL_CG=1
export VLLM_SELF_SPEC_COMPILE_CONSISTENT=1
export VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1
export VLLM_SELF_SPEC_LOG_A2A_COUNTS=1

export W7_MODEL=Qwen/Qwen3-30B-A3B
export W7_TP=1 W7_TRC=0 W7_EAGER=0
export W7_OUTLEN=160 W7_SHORTLEN=32 W7_ITERS=3 W7_WARMUP=2
export W7_GPU_MEM=0.90
export W7_KS=2
export W7_BATCHES=8,32
export W7_OUT="$REPO/research/64_window_e2e/data"
export W7_TAG=q30b_p64
export W7_MASTER_IP=192.168.0.17
export W7_MASTER_PORT=14200
export W7_NODES=2 W7_LOCAL_WORLD=8

# Fail fast on a wedged collective so the retry runner recovers in ~2 min.
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=120
