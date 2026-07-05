# Phase 57 EAGLE env for 2-node runs (sourced on BOTH nodes).
# Copy of research/52_two_node_e2e/scripts/env_2node.sh with the self-spec
# stack (VLLM_SELF_SPEC_*) DROPPED: a real EAGLE3 head must run WITHOUT the
# World A draft flags (cf. Phase 50 STACK=0). Keeps the real-fabric NCCL knobs,
# PATH, HF offline, and model/W7 defaults.
REPO=/h/v-sukmincho/self-spec-moe
export PYTHONPATH="$REPO"
export PATH="$REPO/.venv/bin:$PATH"
export HF_HUB_OFFLINE=1
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0

# REAL fabric: NO forced-PCIe knobs. RoCEv2 rails for NCCL, frontend for
# bootstrap/gloo/DP-RPC.
export NCCL_IB_HCA='^mlx5_8'
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=enmlx0
export GLOO_SOCKET_IFNAME=enmlx0

# NO VLLM_SELF_SPEC_* flags here: EAGLE is a separate drafter model, not the
# self-spec draft path. The worker also guards those env-sets on
# W7_SPEC_METHOD == draft_model.

export W7_MODEL=Qwen/Qwen3-30B-A3B
# W7_EAGER=0 (CUDA graphs): matches Phase 50 and keeps per-step overhead small
# so the verify-comm signal is a large, visible fraction. The DP16 EAGLE spec
# cycle wedges INTERMITTENTLY (first sample_tokens collective) under BOTH cg and
# eager -- eager is not a fix -- so the mitigation is the retry+timeout runner
# (run_ksweep.sh), not enforce_eager.
export W7_TP=1 W7_TRC=0 W7_EAGER=0
export W7_OUTLEN=160 W7_SHORTLEN=32 W7_ITERS=3 W7_WARMUP=2
export W7_GPU_MEM=0.90
export W7_OUT="$REPO/research/57_large_ep_spec_strategy/data"
export W7_TAG=qwen30b_eagle_2node
export W7_MASTER_IP=192.168.0.17
export W7_MASTER_PORT=13355
export W7_NODES=2 W7_LOCAL_WORLD=8

# EAGLE3 drafter selection (real trained head for Qwen3-30B-A3B).
export W7_SPEC_METHOD=eagle3
export W7_SPEC_MODEL=Tengyunw/qwen3_30b_moe_eagle3

# EAGLE auto-enables async scheduling; its batch-queue path deadlocks the
# sample_tokens RPC under DP16 multi-step (K>1) on the 2-node fabric. Force it
# off for a correct, internally-consistent K-sweep (applies to every K).
export W7_ASYNC_SCHED=0

# The EAGLE DP16 spec cycle occasionally deadlocks the first sample_tokens
# collective (intermittent race; not fixed by async-off/eager). Fail fast so a
# wedged run dies in ~120s and the runner retries, instead of the 300s default.
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=120
