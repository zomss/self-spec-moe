# Code Opinion: Forced NIC/Network Emulation in vLLM

Date: 2026-06-23

## Question

Can we edit vLLM to force GPU 6/7 communication through a NIC-like path and use
that as a multi-node EP proxy?

## Short Opinion

Do **not** edit vLLM core to force same-host IB/GDRDMA for this experiment.

The instability is below vLLM: a minimal torch/NCCL all-reduce with
`NCCL_P2P_DISABLE=1`, `NCCL_SHM_DISABLE=1`, and `NCCL_IB_HCA=mlx5_6,mlx5_7`
also fails with:

```text
NET/IB: Got completion ... status=IBV_WC_RETRY_EXC_ERR
```

So changing vLLM's MoE code will not fix the same-host forced-IB failure. The
stable local-network workaround is to force NCCL socket transport, but that is
not close to practical multi-node GPU/IB/DeepEP performance.

The better vLLM-side feature, if we edit code, is **explicit communication-delay
injection for research**, not transport forcing.

## Code Path

For `allgather_reducescatter` EP, the relevant path is:

1. `CudaCommunicator` chooses the all-to-all manager based on
   `parallel_config.all2all_backend`.
2. For `allgather_reducescatter`, it instantiates `AgRsAll2AllManager`.
3. `AgRsAll2AllManager.dispatch()` calls `dist_group.all_gatherv(...)`.
4. `AgRsAll2AllManager.combine()` calls `dist_group.reduce_scatterv(...)`.
5. `GroupCoordinator.all_gatherv()` and `reduce_scatterv()` forward to the
   device communicator.
6. `CudaCommunicator.all_gatherv()` / `reduce_scatterv()` call PyNCCL
   collectives.

Relevant files:

- `vllm/distributed/device_communicators/cuda_communicator.py`
- `vllm/distributed/device_communicators/all2all.py`
- `vllm/distributed/parallel_state.py`
- `vllm/distributed/device_communicators/pynccl.py`

## Why Same-Host Forced IB Is Unstable

The topology says GPU 6 and GPU 7 are NVLink-connected (`NV18`) and have local
NICs (`mlx5_6`, `mlx5_7`) with PIX locality. By setting:

```bash
NCCL_P2P_DISABLE=1
NCCL_SHM_DISABLE=1
NCCL_IB_HCA=mlx5_6,mlx5_7
```

NCCL is forced to avoid NVLink/P2P and use `NET/IB`.

The logs confirm:

```text
NCCL INFO Using network IB
NCCL INFO Channel ... via NET/IB/.../GDRDMA
```

But both vLLM and a minimal torch all-reduce fail/hang. Disabling GDR with:

```bash
NCCL_NET_GDR_LEVEL=0
NCCL_NET_GDR_READ=0
```

still fails with `IBV_WC_RETRY_EXC_ERR`. This makes the problem a same-host
forced-IB/NCCL/IB configuration issue, not a vLLM all-to-all implementation
issue.

## Stable Alternative

The stable local network mode is:

```bash
NCCL_P2P_DISABLE=1
NCCL_SHM_DISABLE=1
NCCL_IB_DISABLE=1
NCCL_SOCKET_IFNAME=enp0s4
NCCL_RAS_ENABLE=0
```

This makes NCCL use:

```text
NET/Socket
```

It completes both:

- minimal torch all-reduce,
- vLLM DP2+EP allgather/reduce-scatter timing.

But it is **not** a practical multi-node IB proxy. It is only a stable way to
disable NVLink/P2P and stress the workflow.

## If We Edit vLLM, What Should We Edit?

### Recommended Research Edit: Delay Injection

Add a research-only delay hook around EP all-to-all, controlled by an
environment variable, for example:

```bash
VLLM_SELF_SPEC_EMULATE_A2A_DELAY_MS=4
```

Best insertion points:

1. `AgRsAll2AllManager.dispatch()` and `combine()`
   - Pros: only affects MoE all-to-all.
   - Best for Self-MoE-spec communication-envelope experiments.
2. `GroupCoordinator.all_gatherv()` and `reduce_scatterv()`
   - Pros: captures all users of these collectives.
   - Cons: too broad; may perturb non-MoE paths.
3. `CudaCommunicator.all_gatherv()` and `reduce_scatterv()`
   - Pros: close to PyNCCL call.
   - Cons: still broader than MoE EP.

I recommend option 1 for research:

```text
AgRsAll2AllManager.dispatch()
AgRsAll2AllManager.combine()
```

This emulates exposed communication on exactly the allgather/reducescatter MoE
EP backend we can run locally.

### What Not To Do

Do not add code whose purpose is to force IB/GDRDMA from inside vLLM. NCCL
transport selection is already controlled by standard NCCL environment variables,
and the failure reproduces outside vLLM.

Also do not claim forced socket results are practical multi-node results.

## Implemented Hook

The research-only delay injection has been added to the MoE EP
allgather/reducescatter path:

```bash
VLLM_SELF_SPEC_EMULATE_A2A_DELAY_MS=<delay_ms>
```

The hook is in `AgRsAll2AllManager` and runs after:

- `dispatch_router_logits()` all-gather,
- `dispatch()` all-gather,
- `combine()` reduce-scatter.

The default is `0`, so normal vLLM behavior is unchanged unless the environment
variable is set.

Important: the delay is **per MoE collective call**, not per decode step. A
model with many MoE layers will apply this delay many times per generated token.
To emulate a target per-step exposed delay, divide that target delay by the
number of delayed MoE collective calls on the decode path.

## Recommended Next Step

Rerun the GPU6/7 DP2+EP workflow with default NCCL transport and:

```text
delay_ms = 0, 0.5, 1, 2, 4, 8
k = 1, 2, 4, 8
```

This would produce a cleaner local emulation than transport forcing:

- stable,
- reproducible,
- directly tied to the all-to-all path,
- honest about being an emulation.

## Bottom Line

The forced-IB instability is not a vLLM bug we should fix for this project. The
useful vLLM edit is a controlled communication-delay emulator in the MoE EP
all-to-all path.
