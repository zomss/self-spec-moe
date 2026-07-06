#!/bin/bash
# Phase 60: rail-engagement smoke. Small nospec b8 run with NCCL_DEBUG=INFO;
# afterwards the 'NET/IB : Using' lines in the logs must list ONLY mlx5_0.
set -u
PHASE=/h/v-sukmincho/self-spec-moe/research/60_one_rail
P52=/h/v-sukmincho/self-spec-moe/research/52_two_node_e2e
P57=/h/v-sukmincho/self-spec-moe/research/57_large_ep_spec_strategy
PY=/h/v-sukmincho/self-spec-moe/.venv/bin/python
ENV="$PHASE/scripts/env_1rail_eagle.sh"
TRY_TO="${1:-900}"
mkdir -p "$PHASE/logs" "$PHASE/data"

for pat in 'w7_2node[.]py' 'EngineCor[e]' 'Worker_D[P]' 'VLLM[:]:'; do
  pkill -9 -f "$pat" 2>/dev/null; ssh h106 "pkill -9 -f '$pat'" 2>/dev/null
done
sleep 5

OVR="W7_NODES=2 W7_LOCAL_WORLD=8 W7_BATCHES=8 W7_ITERS=1 W7_WARMUP=1 \
W7_GPU_MEM=0.90 W7_TAG=q30b_1rail_smoke W7_MASTER_PORT=13860 \
W7_PROMPT_FILE=$P57/data/prompts_ondist.txt W7_CHAT=1 NCCL_DEBUG=INFO"

ssh h106 "bash -c 'source $ENV && export $OVR && W7_NODE_RANK=1 timeout $TRY_TO $PY $P52/scripts/w7_2node.py nospec'" \
    > "$PHASE/logs/smoke_nccl_h106.log" 2>&1 &
H6=$!
( source "$ENV" && export $OVR && W7_NODE_RANK=0 timeout "$TRY_TO" $PY "$P52/scripts/w7_2node.py" nospec ) \
    > "$PHASE/logs/smoke_nccl_h107.log" 2>&1
RC=$?
wait $H6 2>/dev/null
for pat in 'w7_2node[.]py' 'EngineCor[e]' 'Worker_D[P]' 'VLLM[:]:'; do
  pkill -9 -f "$pat" 2>/dev/null; ssh h106 "pkill -9 -f '$pat'" 2>/dev/null
done
echo "smoke RC=$RC"
echo "--- HCA engagement (h107) ---"
grep -h 'NET/IB : Using' "$PHASE/logs/smoke_nccl_h107.log" | sort -u
echo "--- HCA engagement (h106) ---"
grep -h 'NET/IB : Using' "$PHASE/logs/smoke_nccl_h106.log" | sort -u
echo "--- result rows ---"
grep -h 'W7-2N' "$PHASE/logs/smoke_nccl_h107.log"
