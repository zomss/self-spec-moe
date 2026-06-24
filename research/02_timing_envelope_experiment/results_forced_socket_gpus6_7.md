# Results: Stable Forced-Socket Network Path on GPUs 6 and 7

Date: 2026-06-23

## Status

This resolves the forced-network hanging issue for local emulation.

The same-host IB/GDRDMA path selected successfully but hit
`IBV_WC_RETRY_EXC_ERR` during a minimal torch all-reduce. Disabling IB and
forcing NCCL to use the socket network path avoids that failure and completes
both a minimal torch all-reduce probe and a vLLM DP2+EP sweep.

This stable workaround does **not** use IB. It uses NCCL `NET/Socket` over
`enp0s4` with `NCCL_IB_DISABLE=1`. Therefore, it should not be described as
close to a practical multi-node GPU/IB or DeepEP deployment. It is a local
stress test that disables NVLink/P2P, not a practical multi-node proxy.

## Stable NCCL Settings

```bash
CUDA_VISIBLE_DEVICES=6,7
NCCL_P2P_DISABLE=1
NCCL_SHM_DISABLE=1
NCCL_IB_DISABLE=1
NCCL_SOCKET_IFNAME=enp0s4
NCCL_RAS_ENABLE=0
```

Meaning:

- no direct NVLink/P2P transport,
- no host shared-memory transport,
- no IB/GDRDMA transport,
- force NCCL onto the socket network path.

The probe log confirms:

```text
NCCL INFO NET/Socket : Using [0]enp0s4:192.168.0.2
NCCL INFO Using network Socket
NCCL INFO Channel 00/0 : 0[6] -> 1[7] [receive] via NET/Socket/0
```

## vLLM Sweep

Command shape:

```bash
CUDA_VISIBLE_DEVICES=6,7 \
NCCL_P2P_DISABLE=1 \
NCCL_SHM_DISABLE=1 \
NCCL_IB_DISABLE=1 \
NCCL_SOCKET_IFNAME=enp0s4 \
NCCL_RAS_ENABLE=0 \
.venv/bin/python research/02_timing_envelope_experiment/run_dp_latency_sweep.py \
  --model Qwen/Qwen1.5-MoE-A2.7B \
  --load-format dummy \
  --trust-remote-code \
  --data-parallel-size 2 \
  --enable-expert-parallel \
  --all2all-backend allgather_reducescatter \
  --enforce-eager \
  --max-model-len 128 \
  --input-len 1 \
  --output-len 1 \
  --batch-sizes 2,4,6,10
```

## Generated Artifacts

| Artifact | Purpose |
| --- | --- |
| `data/dp2_ep_allgather_forced_socket_gpus6_7_probe.json` | One-iteration forced-socket proof |
| `data/dp2_ep_allgather_forced_socket_gpus6_7.json` | Full forced-socket sweep |
| `data/timings_dp2_ep_forced_socket_gpus6_7_est_f15.csv` | Envelope input |
| `data/envelope_dp2_ep_forced_socket_gpus6_7_est_f15_break_even.csv` | Break-even envelope |
| `data/envelope_dp2_ep_forced_socket_gpus6_7_est_f15_target_1p3.csv` | 1.3x target envelope |
| `logs/forced_socket_torch_probe.log` | Minimal torch forced-socket proof |
| `logs/dp2_ep_allgather_forced_socket_gpus6_7*.log` | vLLM forced-socket logs |

## Measured p50 Latencies

| Global batch size | p50 latency (ms) |
| ---: | ---: |
| 2 | 1560.319 |
| 4 | 1556.714 |
| 6 | 1676.901 |
| 10 | 1603.985 |

## Envelope Results

`t_draft_local_ms` is still estimated with `f_exposed = 0.15`.

| B | k | `S_max` | `beta_min@1.0x` | `beta_min@1.3x` |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 1 | 1.082 | 0.848 | NA |
| 2 | 2 | 1.081 | 0.923 | NA |
| 2 | 4 | 1.129 | 0.939 | NA |

## Interpretation

The hanging issue is resolved by using forced socket networking instead of
forced same-host IB/GDRDMA. However, this remains a local emulation path:

- It proves we can force communication away from NVLink/P2P on GPUs 6 and 7.
- It is stable enough for vLLM DP2+EP timing.
- It does not use IB and does not reproduce real multi-node DeepEP behavior.
- The speedup envelope is still weak at `f_exposed = 0.15`, consistent with all
  prior single-node results.

For future local emulation, use forced socket mode for stable runs and use
analytical delay injection to explore stronger multi-node communication regimes.

For practical multi-node claims, use real IB/DeepEP measurements or validated
multi-node traces. Forced socket results should be reported only as workflow
validation.
