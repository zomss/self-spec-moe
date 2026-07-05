#!/bin/bash
# Phase 57: no-spec Qwen3-30B baseline, 2-node EP16, chat-templated prompts
# (apples-to-apples denominator for the EAGLE speedup). One engine, b8/32/64.
set -u
PHASE=/h/v-sukmincho/self-spec-moe/research/57_large_ep_spec_strategy
P52=/h/v-sukmincho/self-spec-moe/research/52_two_node_e2e
PY=/h/v-sukmincho/self-spec-moe/.venv/bin/python
ENV="$PHASE/scripts/env_eagle_2node.sh"
mkdir -p "$PHASE/logs" "$PHASE/data"

OVR="W7_NODES=2 W7_LOCAL_WORLD=8 W7_BATCHES=8,32,64 W7_ITERS=3 W7_WARMUP=2 \
W7_GPU_MEM=0.90 W7_TAG=q30b_nospec_2n_chat W7_MASTER_PORT=13610 \
W7_PROMPT_FILE=$PHASE/data/prompts_ondist.txt W7_CHAT=1"

for pat in 'w7_2node[.]py' 'EngineCor[e]' 'Worker_D[P]'; do
  pkill -9 -f "$pat" 2>/dev/null; ssh h106 "pkill -9 -f '$pat'" 2>/dev/null
done
sleep 5

ssh h106 "bash -c 'source $ENV && export $OVR && W7_NODE_RANK=1 exec $PY $P52/scripts/w7_2node.py nospec'" \
    > "$PHASE/logs/nospec_2n_chat_h106.log" 2>&1 &
H6=$!
source "$ENV"
export $OVR
W7_NODE_RANK=0 $PY "$P52/scripts/w7_2node.py" nospec \
    > "$PHASE/logs/nospec_2n_chat_h107.log" 2>&1
RC=$?
wait $H6
for pat in 'w7_2node[.]py' 'EngineCor[e]' 'Worker_D[P]'; do
  pkill -9 -f "$pat" 2>/dev/null; ssh h106 "pkill -9 -f '$pat'" 2>/dev/null
done
echo "nospec_2n done RC=$RC"
grep -h 'W7-2N' "$PHASE/logs/nospec_2n_chat_h107.log" | tail -4
