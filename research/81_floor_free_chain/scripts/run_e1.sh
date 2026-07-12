#!/bin/bash
# Phase 81 E1 -- the chain-CG A/B at the miss cell (dense b32/16k, composed
# W4+window draft, K=6). Arms:
#   nospec : fresh baseline
#   pw     : A0 PIECEWISE chain (the 80-E3 config; expect ~3266 tok/s, accept ~5.70)
#   sp     : A1 P69 window-scratchpad FULL-CG chain (VLLM_SELF_SPEC_DRAFT_FULLCG=1;
#            fixed-shape masked-SDPA over sinks+window -- bit-exact greedy claimed)
#   tr     : A2 TRITON_ATTN draft backend + CAPTURED chain graph
#            (W7_SPEC_ATTN_BACKEND=TRITON_ATTN, W7_GQA_FORCE_EAGER_ATTN=0)
#   fa3cg  : A3 CONTROL -- FA3 + forced captured graph = the P35 trap; accept
#            MUST collapse (validates the gate's sensitivity)
# GATE per arm: accept_len within ~2% of pw, THEN compare tok/s.
# GPU: probes for a free one at launch (the box is dynamically shared).
set -u
PHASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$PHASE/../.." && pwd)"
P52="$REPO/research/52_two_node_e2e"
P57="$REPO/research/57_large_ep_spec_strategy"
P76="$REPO/research/76_lever_latency_sweep"
PY="$REPO/.venv/bin/python"
mkdir -p "$PHASE/logs" "$PHASE/data/e1"
ME="$(whoami)"
kill_mine(){ for p in 'w7_2node[.]py' 'EngineCor[e]' 'Worker_D[P]'; do pkill -9 -u "$ME" -f "$p" 2>/dev/null; done; sleep 5; }

pick_gpu(){  # first allowed GPU with >70GB free
  for g in 0 1 6 7; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g")
    [ "$free" -gt 71000 ] && { echo "$g"; return; }
  done
  echo ""; return 1
}

run(){  # arm mode K extra-env...
  local ARM=$1 MODE=$2 K=$3; shift 3
  local GPU; GPU=$(pick_gpu) || { echo "[e1] $ARM: NO FREE GPU -- skipped"; return; }
  local TAG="e1_${ARM}_b32_c16k"
  local LOG="$PHASE/logs/${TAG}.log"
  echo "[e1] $ARM (GPU $GPU, $(date +%H:%M:%S))"
  kill_mine
  ( source "$P76/scripts/env_e76.sh"
    export CUDA_VISIBLE_DEVICES="$GPU"
    export NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo
    unset VLLM_SELF_SPEC_COMPILE_CONSISTENT VLLM_SELF_SPEC_PROFILE
    export VLLM_SELF_SPEC_DRAFT_FULL_CG=1 VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=1
    export VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1 VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1
    export VLLM_SELF_SPEC_SHARED_KV=1
    export W7_DRAFT_QUANT= W7_DRAFT_FULL_REPLICA=0 W7_DRAFT_LOCAL_ROUTE=0 W7_DRAFT_NODE_LOCAL=0
    export W7_MODEL=Qwen/Qwen2.5-7B-Instruct W7_TP=1 W7_EP=0 W7_TRC=0 W7_EAGER=0
    export W7_NODES=1 W7_LOCAL_WORLD=1 W7_MASTER_IP=127.0.0.1
    export W7_GPU_MEM=0.90 W7_OUTLEN=160 W7_SHORTLEN=32
    export W7_CTX_TOKENS=16384 W7_MAX_MODEL_LEN=20480 W7_MAX_NUM_BATCHED=8192
    export W7_KS=$K W7_BATCHES=32 W7_ITERS=4 W7_WARMUP=1
    export W7_TAG="$TAG" W7_OUT="$PHASE/data/e1"
    export W7_PROMPT_FILE="$P57/data/prompts_ondist.txt" W7_CHAT=1
    export W7_MASTER_PORT=$((18100 + RANDOM % 60))
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=180
    if [ "$MODE" = spec ]; then
      export W7_SPEC_METHOD=draft_model W7_SPEC_MODEL="$E76_W4"
      export VLLM_DISABLED_KERNELS="MacheteLinearKernel,CutlassW4A8LinearKernel,AllSparkLinearKernel"
      export VLLM_SELF_SPEC_DRAFT_KV_WINDOW=512 VLLM_SELF_SPEC_DRAFT_KV_SINKS=16
    fi
    for kv in "$@"; do export "${kv?}"; done
    W7_NODE_RANK=0 timeout 2400 "$PY" "$P52/scripts/w7_2node.py" "$MODE"
  ) > "$LOG" 2>&1
  grep -hE 'W7-2N.*batch=' "$LOG" | tail -1 | sed 's/^/       /'
  grep -hE 'Draft FULL-CG|FULLCG|scratchpad|TRITON' "$LOG" | head -2 | sed 's/^/       /'
  kill_mine
}

FILTER="${1:-.}"
want(){ echo "$1" | grep -qE "$FILTER"; }
want nospec && run nospec nospec 0
want pw     && run pw     spec 6
want sp     && run sp     spec 6 VLLM_SELF_SPEC_DRAFT_FULLCG=1
want spk4   && run spk4   spec 4 VLLM_SELF_SPEC_DRAFT_FULLCG=1
want sp0    && run sp0    spec 6 VLLM_SELF_SPEC_DRAFT_FULLCG=1 VLLM_SELF_SPEC_DRAFT_STEP0_FULL_CG=1
want spla   && run spla   spec 6 VLLM_SELF_SPEC_DRAFT_FULLCG=1 VLLM_SELF_SPEC_DRAFT_STEP0_FULL_CG=1 W7_LOCAL_ARGMAX=1
want spas   && run spas   spec 6 VLLM_SELF_SPEC_DRAFT_FULLCG=1 VLLM_SELF_SPEC_DRAFT_STEP0_FULL_CG=1 W7_ASYNC_SCHED=1
want spco   && run spco   spec 6 VLLM_SELF_SPEC_DRAFT_FULLCG=1 VLLM_SELF_SPEC_DRAFT_STEP0_FULL_CG=1 VLLM_SELF_SPEC_CPU_ORCH=1
want spcoas && run spcoas spec 6 VLLM_SELF_SPEC_DRAFT_FULLCG=1 VLLM_SELF_SPEC_DRAFT_STEP0_FULL_CG=1 VLLM_SELF_SPEC_CPU_ORCH=1 W7_ASYNC_SCHED=1
want spsd   && run spsd   spec 6 VLLM_SELF_SPEC_DRAFT_FULLCG=1 VLLM_SELF_SPEC_DRAFT_STEP0_FULL_CG=1 VLLM_SELF_SPEC_CPU_ORCH=1 W7_ASYNC_SCHED=1 VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE=1
want spsd2  && run spsd2  spec 6 VLLM_SELF_SPEC_CPU_ORCH=1 W7_ASYNC_SCHED=1 VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE=1
want spdbg  && run spdbg  spec 6 VLLM_SELF_SPEC_DRAFT_FULLCG=1 VLLM_SELF_SPEC_DRAFT_STEP0_FULL_CG=1 VLLM_SELF_SPEC_CPU_ORCH=1 W7_ASYNC_SCHED=1 VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE=1 W7_STEP0_DEBUG=1
want spw0   && run spw0   spec 6 VLLM_SELF_SPEC_CPU_ORCH=1 W7_ASYNC_SCHED=1 VLLM_SELF_SPEC_DRAFT_KV_WINDOW=0 VLLM_SELF_SPEC_DRAFT_KV_SINKS=0
want spw0sd && run spw0sd spec 6 VLLM_SELF_SPEC_CPU_ORCH=1 W7_ASYNC_SCHED=1 VLLM_SELF_SPEC_DRAFT_KV_WINDOW=0 VLLM_SELF_SPEC_DRAFT_KV_SINKS=0 VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE=1
want spsdfix && run spsdfix spec 6 VLLM_SELF_SPEC_DRAFT_FULLCG=1 VLLM_SELF_SPEC_DRAFT_STEP0_FULL_CG=1 VLLM_SELF_SPEC_CPU_ORCH=1 W7_ASYNC_SCHED=1 VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE=1
want spall  && run spall  spec 6 VLLM_SELF_SPEC_DRAFT_FULLCG=1 VLLM_SELF_SPEC_DRAFT_STEP0_FULL_CG=1 W7_LOCAL_ARGMAX=1 W7_ASYNC_SCHED=1
want tr     && run tr     spec 6 W7_SPEC_ATTN_BACKEND=TRITON_ATTN W7_GQA_FORCE_EAGER_ATTN=0 VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=0
want fa3cg  && run fa3cg  spec 6 W7_GQA_FORCE_EAGER_ATTN=0 VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE=0
echo "[e1] DONE ($(date +%H:%M:%S))"
