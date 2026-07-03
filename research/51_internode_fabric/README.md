# Phase 51: internode_fabric — raw h107↔h106 fabric microbench (the Phase 19 gate)

**Source phases:** 19 (multinode plan: go/no-go gate = exposed inter-node A2A
>= ~200us/collective), 42 (decode A2A is latency-bound, 16-64 KB/rank/step;
realistic good-fabric collective 30-200us), 47 (win curve vs a2a us/coll:
measured-shielded 1.19x@100us -> 1.95x@1000us at b64; projection 1.6-2.1x at
f~0.6-0.8).

**Objective.** First real inter-node measurement: h107 + h106 (2x 8xH100, eight
400G ConnectX-7 RoCEv2 rails each, GPUDirect RDMA, shared NFS home). Measure
(1) raw per-rail RDMA bandwidth/latency (`ib_write_bw`/`ib_write_lat`, GID
index 3 = RoCEv2/IPv4), (2) NCCL `all_to_all_single` and small `all_reduce`
per-collective latency across 2 nodes x 8 GPUs at decode-sized payloads
(16 KB - 64 MB total send buffer per rank), through the same torch-2.11 /
NCCL-2.28.9 stack vLLM uses.

**Assumptions.** Rails are 192.168.{0..7}.{17=h107,16=h106}; frontend
ens14np0 (10.31.199.x) used only for torchrun rendezvous. `nvidia_peermem` +
`gdrdrv` loaded on both nodes (verified). No other GPU jobs running.

**Commands.**
- `scripts/run_perftest.sh` — raw RDMA per rail.
- `scripts/bench_nccl_a2a.py` via `scripts/run_nccl_bench.sh` — launches
  torchrun on both nodes (h106 side over SSH), NCCL_IB_HCA=mlx5_0..7,
  NCCL_IB_GID_INDEX=3.

**Decision criteria.** Map measured per-collective A2A latency at decode sizes
onto the Phase 47 win curve: >=~200us/coll -> Phase 19 GO (predicted >=1.2-1.4x,
Phase 47 shielded ~1.4-1.6x); 30-100us -> modest-win regime (~1.1-1.4x at
b32-64); also record whether NCCL uses GPUDirect RDMA on this RoCE fabric.

**Expected next artifact.** `results_fabric.md` with the measured latency
table + go/no-go reading; feeds the first real 2-node vLLM run (Phase 52).
