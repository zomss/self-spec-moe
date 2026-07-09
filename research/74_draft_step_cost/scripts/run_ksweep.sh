#!/bin/bash
# Clean K-sweep to isolate the per-draft-step cost (forward + sample + plumbing)
# from fixed cost (verify + scheduler). base spec, 16k b8, NO profiler,
# batch-invariant OFF. cycle(K) = B*accept(K)/tok_s(K) = C_fixed + K*C_step.
# If C_step >> pure draft forward, there is per-step (sample) overhead to cut.
set -u
PHASE=/data/smcho/self-spec-moe/research/74_draft_step_cost
P52=/data/smcho/self-spec-moe/research/52_two_node_e2e
P57=/data/smcho/self-spec-moe/research/57_large_ep_spec_strategy
PY=/data/smcho/self-spec-moe/.venv/bin/python
ME="$(whoami)"
kill_mine(){ for p in 'w7_2node[.]py' 'EngineCor[e]' 'Worker_D[P]'; do pkill -9 -u "$ME" -f "$p" 2>/dev/null; done; sleep 5; }
for K in 1 2 4 6; do
  LOG="$PHASE/logs/ks_K${K}_16k_b8.log"
  kill_mine
  ( source "$PHASE/scripts/env_cloud4.sh"
    unset VLLM_SELF_SPEC_COMPILE_CONSISTENT
    export W7_KS=$K W7_BATCHES=8 W7_ITERS=3 W7_WARMUP=1 W7_TAG="p74_ks_K${K}_16k_b8" \
      W7_MASTER_PORT=$((17200 + K + RANDOM%30)) W7_CTX_TOKENS=16384 W7_MAX_MODEL_LEN=20480 \
      W7_MAX_NUM_BATCHED=8192 W7_OUT="$PHASE/data" W7_PROMPT_FILE="$P57/data/prompts_ondist.txt" W7_CHAT=1
    W7_NODE_RANK=0 timeout 1200 "$PY" "$P52/scripts/w7_2node.py" spec ) > "$LOG" 2>&1
  echo "[ks] K=$K: $(grep -hE 'W7-2N.*batch=' "$LOG" | tail -1)"
  kill_mine
done
echo "[ks] DONE ($(date +%H:%M:%S))"
