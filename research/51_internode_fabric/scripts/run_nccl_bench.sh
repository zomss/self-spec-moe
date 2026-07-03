#!/bin/bash
# 2-node NCCL bench launcher: run on h107, starts h106 side over SSH.
set -u
PHASE=/h/v-sukmincho/self-spec-moe/research/51_internode_fabric
PY=/h/v-sukmincho/self-spec-moe/.venv/bin/torchrun
MASTER=10.31.199.17
PORT=29617

ENVS="NCCL_SOCKET_IFNAME=ens14np0 NCCL_IB_HCA=^mlx5_8 NCCL_IB_GID_INDEX=3 \
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET"

COMMON="--nnodes=2 --nproc-per-node=8 --master-addr=$MASTER --master-port=$PORT \
$PHASE/scripts/bench_nccl_a2a.py"

ssh h106 "cd $PHASE && env $ENVS $PY --node-rank=1 $COMMON" \
    > "$PHASE/logs/nccl_h106.log" 2>&1 &
H106_PID=$!

env $ENVS $PY --node-rank=0 $COMMON > "$PHASE/logs/nccl_h107.log" 2>&1
RC=$?
wait $H106_PID
exit $RC
