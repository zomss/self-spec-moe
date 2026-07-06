#!/bin/bash
# Phase 61: 1-rail (NCCL_IB_HCA='=mlx5_0') 2-node DeepSeek-V2 236B runner.
# TP2 x DP4-per-node = EP16, the Phase 53/54 known-good engine recipe
# (run_ladder.sh step3 at gpu_mem 0.95 per run_step3_095.sh: 0.90 leaves
# 2-3k KV tokens/rank on HEAD -> preemption thrash + timing hang at b>=32).
# Runner pattern = Phase 60 run_1rail.sh: ONE engine invocation per arm under
# a hard `timeout`, GPU-pid force-kill between tries on BOTH nodes, retry on
# the intermittent DP first-collective wedge. 236B + 1 rail is slow ->
# generous default try timeout.
# Usage: run_1rail_236b.sh {nospec|worldA} [batches] [retries] [try_to_s]
# Env: NCCLDBG=1 adds NCCL_DEBUG=INFO (rail-engagement check).
set -u
ARM="${1:?usage: run_1rail_236b.sh nospec|worldA [batches] [retries] [try_to_s]}"
BATCHES="${2:-8,32,64}"
RETRIES="${3:-3}"
TRY_TO="${4:-1800}"
PHASE=/h/v-sukmincho/self-spec-moe/research/61_236b_one_rail
P52=/h/v-sukmincho/self-spec-moe/research/52_two_node_e2e
PY=/h/v-sukmincho/self-spec-moe/.venv/bin/python
ENV="$PHASE/scripts/env_1rail_2node.sh"
mkdir -p "$PHASE/logs" "$PHASE/data"

# Phase 53/54 known-good 236B engine settings; default synthetic prompts
# (NO W7_PROMPT_FILE/W7_CHAT) to match the Phase 53/54 8-rail references.
COMMON="W7_MODEL=deepseek-ai/DeepSeek-V2 W7_TRC=1 \
W7_NODES=2 W7_LOCAL_WORLD=4 W7_TP=2 W7_ITERS=2 W7_WARMUP=1 \
W7_GPU_MEM=0.95 W7_MAX_MODEL_LEN=1024 W7_CG_SIZES=8,16,32,64,128 \
W7_MAX_NUM_BATCHED=2048 W7_OUT=$PHASE/data"

case "$ARM" in
  nospec)
    MODE=nospec; K=0; TAG=dsv2_1rail_nospec; PORT=13900
    OVR="$COMMON" ;;
  worldA)
    # Node-local draft (SP/TP2 node-gather branch), K=1, no full replica
    # (does not fit at 236B). Coord-skip/amortize/step0CG explicitly OFF
    # (= defaults; matches the 8-rail reference run_step3_095.sh).
    MODE=spec; K=1; TAG=dsv2_1rail_worldA; PORT=13950
    OVR="$COMMON W7_KS=1 W7_DRAFT_LOCAL_ROUTE=0 W7_DRAFT_NODE_LOCAL=1 \
W7_DRAFT_FULL_REPLICA=0 VLLM_SELF_SPEC_DRAFT_SKIP_DP_COORD=0 \
VLLM_SELF_SPEC_DRAFT_AMORTIZE_DP_COORD=0 \
VLLM_SELF_SPEC_DRAFT_STEP0_FULL_CG=0" ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac

kill_stragglers() {
  for pat in 'w7_2node[.]py' 'EngineCor[e]' 'Worker_D[P]' 'VLLM[:]:'; do
    pkill -9 -f "$pat" 2>/dev/null
    ssh h106 "pkill -9 -f '$pat'" 2>/dev/null
  done
  # wedged NCCL workers ignore pkill: force-kill by GPU pid on both nodes.
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u); do
    kill -9 "$p" 2>/dev/null
  done
  ssh h106 'for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u); do kill -9 "$p" 2>/dev/null; done' 2>/dev/null
  sleep 8
}

if [ "$MODE" = "nospec" ]; then
  ROWPAT="W7-2N nospec\] batch=.* tok/s"
else
  ROWPAT="W7-2N K=$K\] batch=.* accept_len"
fi
nb=$(echo "$BATCHES" | awk -F, '{print NF}')

ok=0
for try in $(seq 1 "$RETRIES"); do
  kill_stragglers
  LOGH="$PHASE/logs/${ARM}_try${try}.log"
  LOG6="$PHASE/logs/${ARM}_try${try}_h106.log"
  RUN="$OVR W7_BATCHES=$BATCHES W7_TAG=$TAG \
W7_MASTER_PORT=$((PORT+try*7)) ${NCCLDBG:+NCCL_DEBUG=INFO}"
  echo "[1rail-236b] $ARM try=$try to=${TRY_TO}s ($(date +%H:%M:%S))"
  ssh h106 "bash -c 'source $ENV && export $RUN && W7_NODE_RANK=1 timeout $TRY_TO $PY $P52/scripts/w7_2node.py $MODE'" \
      > "$LOG6" 2>&1 &
  H6=$!
  ( source "$ENV" && export $RUN && W7_NODE_RANK=0 timeout "$TRY_TO" $PY "$P52/scripts/w7_2node.py" "$MODE" ) \
      > "$LOGH" 2>&1
  wait "$H6" 2>/dev/null
  got=$(grep -cE "$ROWPAT" "$LOGH" 2>/dev/null)
  echo "[1rail-236b] $ARM try=$try -> ${got:-0}/$nb batch rows ($(date +%H:%M:%S))"
  grep -hE "W7-2N" "$LOGH" 2>/dev/null
  if [ "${got:-0}" -ge "$nb" ]; then ok=1; break; fi
done
[ "$ok" = "1" ] || echo "[1rail-236b] $ARM INCOMPLETE after $RETRIES tries"
kill_stragglers
echo "[1rail-236b] $ARM done ($(date +%H:%M:%S))"
