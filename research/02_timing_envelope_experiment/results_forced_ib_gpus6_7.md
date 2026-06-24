# Results: Forcing GPU 6/7 Communication Through IB

Date: 2026-06-23

## Question

Can GPUs 6 and 7 be forced to communicate through a NIC or NIC-like stack to
emulate multi-node EP?

## Short Answer

**Partially yes.** On this node, GPU 6 is PIX-local to `mlx5_6` and GPU 7 is
PIX-local to `mlx5_7`. Both IB ports are active. By disabling NCCL P2P and shared
memory, NCCL can be forced to select the IB/GDRDMA transport between GPUs 6 and
7.

However, this is still not a true multi-node setup. It is same-host IB/GDRDMA
traffic and may not reproduce real cross-node congestion, routing, switch hops,
or DeepEP behavior. In both the vLLM DP2+EP probe and a minimal torch
all-reduce probe, NCCL selected IB/GDRDMA but the path failed or hung with an
IB retry error. The stable workaround is to force NCCL onto the socket network
path instead; see `results_forced_socket_gpus6_7.md`.

That workaround does not use IB and should not be treated as close to practical
multi-node IB/DeepEP performance. It is only a stable local way to avoid
NVLink/P2P.

## Topology Evidence

`nvidia-smi topo -m` shows:

| Pair | Topology |
| --- | --- |
| GPU6 -> NIC6 (`mlx5_6`) | PIX |
| GPU7 -> NIC7 (`mlx5_7`) | PIX |
| GPU6 -> GPU7 | NV18 |

IB link state:

| NIC | State | Physical state | Link layer |
| --- | --- | --- | --- |
| `mlx5_6` | ACTIVE | LinkUp | InfiniBand |
| `mlx5_7` | ACTIVE | LinkUp | InfiniBand |

## Forcing NCCL to Use IB

The probe used:

```bash
CUDA_VISIBLE_DEVICES=6,7 \
NCCL_DEBUG=INFO \
NCCL_DEBUG_SUBSYS=INIT,NET \
NCCL_P2P_DISABLE=1 \
NCCL_SHM_DISABLE=1 \
NCCL_IB_HCA=mlx5_6,mlx5_7 \
.venv/bin/python research/02_timing_envelope_experiment/run_dp_latency_sweep.py ...
```

Important settings:

- `NCCL_P2P_DISABLE=1`: prevents direct GPU P2P/NVLink transport.
- `NCCL_SHM_DISABLE=1`: prevents host shared-memory transport.
- `NCCL_IB_HCA=mlx5_6,mlx5_7`: restricts NCCL to the NICs local to GPU 6/7.

## NCCL Log Evidence

The log confirms:

```text
NCCL INFO NET/IB : Using [0]mlx5_6:1/IB [1]mlx5_7:1/IB
NCCL INFO Using network IB
NCCL INFO NCCL_P2P_DISABLE set by environment to 1
NCCL INFO Channel 00/0 : 0[6] -> 1[7] [receive] via NET/IB/1/GDRDMA
NCCL INFO Channel 00/0 : 1[7] -> 0[6] [send] via NET/IB/1/GDRDMA
```

So the transport was successfully forced away from NVLink/P2P and onto
IB/GDRDMA.

## Current Limitation

The vLLM forced-IB probe did not complete. It initialized NCCL and selected
IB/GDRDMA, but then hung before writing the output JSON. A minimal torch
all-reduce reproduced the lower-level problem:

```text
NCCL WARN NET/IB: Got completion ... status=IBV_WC_RETRY_EXC_ERR
```

The process was stopped to release GPUs 6 and 7.

Therefore:

- **valid:** transport selection proof,
- **not valid yet:** latency benchmark result.

## Recommended Use

Do not use forced same-host IB/GDRDMA for stable Phase 02 timing on this
machine. Use forced socket networking instead:

```bash
NCCL_P2P_DISABLE=1
NCCL_SHM_DISABLE=1
NCCL_IB_DISABLE=1
NCCL_SOCKET_IFNAME=enp0s4
NCCL_RAS_ENABLE=0
```

Keep forced-IB only as evidence that NCCL can select the NIC/GDRDMA path; it is
not stable enough here for timing.
