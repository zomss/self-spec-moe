#!/bin/bash
# Phase 60: 1-rail (NCCL_IB_HCA='=mlx5_0') 2-node DP16/EP16 runner, all arms.
# Clone of research/57_large_ep_spec_strategy/scripts/run_ksweep.sh: ONE engine
# invocation per arm wrapped in a hard `timeout`, GPU-pid force-kill between
# tries on BOTH nodes, retry on the intermittent DP16 first-collective wedge.
# 1-rail runs are comm-slow -> generous default try timeout.
# Usage: run_1rail.sh {nospec|eagleK1|eagleK2|worldA} [batches] [retries] [try_to_s]
# Env: NCCLDBG=1 adds NCCL_DEBUG=INFO (rail-engagement check).
set -u
ARM="${1:?usage: run_1rail.sh nospec|eagleK1|eagleK2|worldA [batches] [retries] [try_to_s]}"
BATCHES="${2:-8,32,64}"
RETRIES="${3:-4}"
TRY_TO="${4:-900}"
PHASE=/h/v-sukmincho/self-spec-moe/research/60_one_rail
P52=/h/v-sukmincho/self-spec-moe/research/52_two_node_e2e
P57=/h/v-sukmincho/self-spec-moe/research/57_large_ep_spec_strategy
PY=/h/v-sukmincho/self-spec-moe/.venv/bin/python
mkdir -p "$PHASE/logs" "$PHASE/data"

case "$ARM" in
  nospec)  ENV="$PHASE/scripts/env_1rail_eagle.sh";  MODE=nospec; K=0; TAG=q30b_1rail_nospec; PORT=13700 ;;
  eagleK1) ENV="$PHASE/scripts/env_1rail_eagle.sh";  MODE=spec;   K=1; TAG=q30b_1rail_eagle;  PORT=13740 ;;
  eagleK2) ENV="$PHASE/scripts/env_1rail_eagle.sh";  MODE=spec;   K=2; TAG=q30b_1rail_eagle;  PORT=13770 ;;
  worldA)  ENV="$PHASE/scripts/env_1rail_worldA.sh"; MODE=spec;   K=2; TAG=q30b_1rail_worldA; PORT=13820 ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac

kill_stragglers() {
  for pat in 'w7_2node[.]py' 'EngineCor[e]' 'Worker_D[P]' 'VLLM[:]:'; do
    pkill -9 -f "$pat" 2>/dev/null
    ssh h106 "pkill -9 -f '$pat'" 2>/dev/null
  done
  # NOTE (2026-07-05): shared nodes -- cleanup is pkill of OUR patterns
  # only (own-user). Never kill arbitrary GPU PIDs / other users' jobs.
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
  OVR="W7_NODES=2 W7_LOCAL_WORLD=8 W7_KS=$K W7_BATCHES=$BATCHES \
W7_ITERS=3 W7_WARMUP=2 W7_GPU_MEM=0.90 W7_TAG=$TAG W7_MASTER_PORT=$((PORT+try*7)) \
W7_PROMPT_FILE=$P57/data/prompts_ondist.txt W7_CHAT=1 ${NCCLDBG:+NCCL_DEBUG=INFO}"
  echo "[1rail] $ARM try=$try to=${TRY_TO}s ($(date +%H:%M:%S))"
  ssh h106 "bash -c 'source $ENV && export $OVR && W7_NODE_RANK=1 timeout $TRY_TO $PY $P52/scripts/w7_2node.py $MODE'" \
      > "$LOG6" 2>&1 &
  H6=$!
  ( source "$ENV" && export $OVR && W7_NODE_RANK=0 timeout "$TRY_TO" $PY "$P52/scripts/w7_2node.py" "$MODE" ) \
      > "$LOGH" 2>&1
  wait "$H6" 2>/dev/null
  got=$(grep -cE "$ROWPAT" "$LOGH" 2>/dev/null)
  echo "[1rail] $ARM try=$try -> ${got:-0}/$nb batch rows ($(date +%H:%M:%S))"
  grep -hE "W7-2N" "$LOGH" 2>/dev/null
  if [ "${got:-0}" -ge "$nb" ]; then ok=1; break; fi
done
[ "$ok" = "1" ] || echo "[1rail] $ARM INCOMPLETE after $RETRIES tries"
kill_stragglers
echo "[1rail] $ARM done ($(date +%H:%M:%S))"
