#!/bin/bash
# Phase 56 Stage A: 2-node overlap microbench launcher (16 ranks).
set -u
PHASE=/h/v-sukmincho/self-spec-moe/research/56_overlap_node_local
PY=/h/v-sukmincho/self-spec-moe/.venv/bin/torchrun
MASTER=10.31.199.17
PORT=29645
mkdir -p "$PHASE/logs"

ENVS="NC=${NC:-30} NM=${NM:-40} NCCL_SOCKET_IFNAME=ens14np0 NCCL_IB_HCA=^mlx5_8 NCCL_IB_GID_INDEX=3"
COMMON="--nnodes=2 --nproc-per-node=8 --master-addr=$MASTER --master-port=$PORT \
$PHASE/scripts/overlap_microbench.py"

for pat in 'microbench[.]py' 'torchru[n]'; do
  pkill -9 -f "$pat" 2>/dev/null; ssh h106 "pkill -9 -f '$pat'" 2>/dev/null
done
sleep 3

ssh h106 "cd $PHASE && env $ENVS $PY --node-rank=1 $COMMON" \
    > "$PHASE/logs/microbench_h106.log" 2>&1 &
H106_PID=$!
env $ENVS $PY --node-rank=0 $COMMON > "$PHASE/logs/microbench_h107.log" 2>&1
RC=$?
wait $H106_PID
echo "microbench RC=$RC"
grep -E 'T_comm|T_compute|T_concurrent|max\(|overlap_ratio|HIDDEN|world=' "$PHASE/logs/microbench_h107.log"
