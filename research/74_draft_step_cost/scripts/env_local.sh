# Phase 74 single-node env, PORTABLE across boxes (supersedes env_cloud4.sh,
# which hardcodes /data/smcho + GPUs 4-7 for cloud-9ezI3Q).
#
# Deltas vs env_cloud4.sh -- behavior is otherwise byte-identical:
#   - REPO derived from this script's own path (works on any checkout).
#   - GPUs via ${W7_GPUS:-4,5,6,7}; DP size follows the GPU count.
# Everything else (the Phase-47/52 self-spec stack, Phase-65 systems fixes,
# Phase-66 SHARED_KV, EP-routed bf16 base draft) is unchanged.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH="$REPO"
export PATH="$REPO/.venv/bin:$PATH"
export HF_HUB_OFFLINE=1
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0

# Which GPUs to use; DP width = how many were named.
export CUDA_VISIBLE_DEVICES="${W7_GPUS:-4,5,6,7}"
_NGPU=$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")

# Single node: loopback for DP-RPC / NCCL+gloo bootstrap (docker bridges present
# -> force lo so auto-select can't pick 172.18.x and hang).
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
unset NCCL_IB_HCA NCCL_IB_GID_INDEX

# Phase-47/52 self-spec stack (proven): step-0 full-CG + PIECEWISE chain (eager
# per-step metadata rebuild -- required for the window-KV draft to engage).
export VLLM_SELF_SPEC_DRAFT_FULL_CG=1
export VLLM_SELF_SPEC_COMPILE_CONSISTENT=1
export VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1

# Phase-65 systems fixes (valid for the EP-routed draft; SKIP_DP_COORD is NOT
# set -- it requires a comm-free draft, which we are deliberately not using).
export VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1
export VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1
# Phase-66 residency win (systems, always on): draft binds target KV tensors.
export VLLM_SELF_SPEC_SHARED_KV=1

# Base arm draft = EP-routed bf16 (no quant, no replica, no local routing).
export W7_DRAFT_QUANT=
export W7_DRAFT_FULL_REPLICA=0
export W7_DRAFT_LOCAL_ROUTE=0
export W7_DRAFT_NODE_LOCAL=0

export W7_MODEL=Qwen/Qwen3-30B-A3B
export W7_TP=1 W7_TRC=0 W7_EAGER=0
export W7_OUTLEN=160 W7_SHORTLEN=32
export W7_GPU_MEM=0.90
export W7_MASTER_IP=127.0.0.1
export W7_NODES=1 W7_LOCAL_WORLD="$_NGPU"

# Fail fast on a wedged collective (~2 min) so the retry runner recovers.
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=120
