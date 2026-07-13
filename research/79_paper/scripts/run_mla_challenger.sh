#!/bin/bash
# OFF-hardening measurement: the top challenger from off_hardening.md --
# MLA (DeepSeek-V2-Lite) b32/16k, pure q_fp8 self-draft (no window), priced
# 1.13x optimistic at gamma7. First MLA e2e in the record.
# TP1 single-GPU (16B fits; matches the NVLink-muted-comm regime of the OFF
# claim). No window => no scratchpad chain graphs; floor mitigation uses the
# window-independent knobs (step0 CG, CPU_ORCH, async sched, compacted
# step-0 -- the E2b fix handles the full-KV path via clone).
# Run BY PATH: bash scripts/run_mla_challenger.sh
set -u
PHASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$PHASE/../.." && pwd)"
P52="$REPO/research/52_two_node_e2e"
P57="$REPO/research/57_large_ep_spec_strategy"
PY="$REPO/.venv/bin/python"
mkdir -p "$PHASE/logs" "$PHASE/data/mla_challenger"
ME="$(whoami)"
kill_mine(){ for p in 'w7_2node[.]py' 'EngineCor[e]' 'Worker_D[P]'; do pkill -9 -u "$ME" -f "$p" 2>/dev/null; done; sleep 5; }

run(){  # gpu arm mode K [extra-env...]
  local GPU=$1 ARM=$2 MODE=$3 K=$4 CTX=16384; shift 4
  local TAG="mlach_${ARM}_K${K}_b32_c16k"
  local LOG="$PHASE/logs/${TAG}.log"
  echo "[mlach] $TAG (GPU $GPU, $(date +%H:%M:%S))"
  kill_mine
  ( export PATH="$REPO/.venv/bin:$PATH"   # MLA JIT needs ninja
    export HF_HOME=/data/smcho/huggingface HF_HUB_OFFLINE=1 TMPDIR=/data/smcho/tmp
    export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
    export CUDA_VISIBLE_DEVICES="$GPU"
    export NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo
    unset VLLM_SELF_SPEC_COMPILE_CONSISTENT VLLM_SELF_SPEC_PROFILE
    export VLLM_SELF_SPEC_DRAFT_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1
    export VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1
    export VLLM_SELF_SPEC_SHARED_KV=1
    export W7_DRAFT_FULL_REPLICA=0 W7_DRAFT_LOCAL_ROUTE=0 W7_DRAFT_NODE_LOCAL=0
    export W7_MODEL=deepseek-ai/DeepSeek-V2-Lite W7_TP=1 W7_EP=0 W7_TRC=0 W7_EAGER=0
    export W7_NODES=1 W7_LOCAL_WORLD=1 W7_MASTER_IP=127.0.0.1
    export W7_GPU_MEM=0.90 W7_OUTLEN=160 W7_SHORTLEN=32
    export W7_CTX_TOKENS=$CTX W7_MAX_MODEL_LEN=$((CTX + 4096)) W7_MAX_NUM_BATCHED=8192
    export W7_KS=$K W7_BATCHES=32 W7_ITERS=4 W7_WARMUP=1
    export W7_TAG="$TAG" W7_OUT="$PHASE/data/mla_challenger"
    export W7_PROMPT_FILE="$P57/data/prompts_ondist.txt" W7_CHAT=1
    export W7_MASTER_PORT=$((18700 + K + RANDOM % 40))
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=180
    if [ "$MODE" = spec ]; then
      export W7_DRAFT_QUANT=fp8_per_block     # the q_fp8 challenger (native)
      # floor mitigation, window-independent subset of the 81 stack:
      export VLLM_SELF_SPEC_CPU_ORCH=1 W7_ASYNC_SCHED=1
      export VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE=1
    fi
    for kv in "$@"; do export "${kv?}"; done
    W7_NODE_RANK=0 timeout 2400 "$PY" "$P52/scripts/w7_2node.py" "$MODE"
  ) > "$LOG" 2>&1
  grep -hE 'W7-2N.*batch=' "$LOG" | tail -1 | sed "s/^/       [$TAG] /"
  kill_mine
}

FILTER="${1:-.}"
want(){ echo "$1" | grep -qE "$FILTER"; }
want nospec && run 0 nospec nospec 0
want specK7 && run 0 spec   spec   7
want specK5 && run 0 spec   spec   5
# anomaly chase: bf16 self-draft = draft-path sanity (expect accept ~K+1);
# fp8_per_block is W8A8 (acts too) -- offline beta was WEIGHT-ONLY
want bf16K5 && run 0 bf16   spec   5 W7_DRAFT_QUANT=
# capture hypothesis (F11 on FLASH_ATTN_MLA): eager chain attention
want bf16eager && run 0 bf16eager spec 5 W7_DRAFT_QUANT= W7_GQA_FORCE_EAGER_ATTN=1
# backend hypothesis: fg2 validated replay-safety on FLASHMLA, not FA3-MLA
want bf16fmla && run 0 bf16fmla spec 5 W7_DRAFT_QUANT= W7_SPEC_ATTN_BACKEND=FLASHMLA
kill_mine
echo "[mlach] DONE ($(date +%H:%M:%S))"
