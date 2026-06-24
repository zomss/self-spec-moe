# Results: Calibrated Runtime Delay Sweep on GPUs 6 and 7

Date: 2026-06-23

## Status

This run uses the research-only runtime hook:

```bash
VLLM_SELF_SPEC_EMULATE_A2A_DELAY_MS=<delay_ms>
```

with default NCCL transport on GPUs 6 and 7. The hook injects host-side delay
after the MoE EP allgather/reducescatter path for `allgather_reducescatter`.

This is a local emulation. It is not a real multi-node DeepEP/DBO run.

## Sweep

| Setting | Value |
| --- | --- |
| GPUs | `CUDA_VISIBLE_DEVICES=6,7` |
| Model config | `Qwen/Qwen1.5-MoE-A2.7B` |
| Weights | `--load-format dummy` |
| Parallelism | `--data-parallel-size 2 --enable-expert-parallel` |
| Backend | `allgather_reducescatter` |
| Decode shape | `input_len=1`, `output_len=1` |
| Batch sizes | `2,4,6,10` |
| Delay values | `0, 0.025, 0.05, 0.1, 0.2 ms` |

## Generated Artifacts

| Artifact | Purpose |
| --- | --- |
| `data/dp2_ep_delay_*ms_gpus6_7.json` | Raw runtime latency sweeps |
| `data/delay_sweep_latencies_gpus6_7.csv` | Extracted p50 latencies |
| `data/timings_delay_sweep_gpus6_7_skip_delay.csv` | Envelope input using no-delay draft proxy |
| `data/envelope_delay_sweep_gpus6_7_skip_delay_break_even.csv` | Break-even envelope |
| `data/envelope_delay_sweep_gpus6_7_skip_delay_target_1p3.csv` | 1.3x target envelope |
| `data/acceptance_sweep_delay_gpus6_7_skip_delay.csv` | Acceptance sweep |

## Measured p50 Latencies

| Delay (ms) | B=2 | B=4 | B=6 | B=10 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1675.289 | 1697.175 | 1734.891 | 1570.359 |
| 0.025 | 1822.545 | 1887.519 | 1777.244 | 2078.587 |
| 0.05 | 1764.694 | 1687.273 | 1772.432 | 2202.849 |
| 0.1 | 1797.993 | 1886.093 | 1778.077 | 2074.621 |
| 0.2 | 1980.262 | 2018.044 | 1987.217 | 1966.275 |

## Envelope Method

For this runtime sweep, the envelope uses:

```text
T_base_ms = measured latency with delay enabled
T_draft_local_ms = measured no-delay latency at the same B
T_verify_ms = measured latency with delay enabled at B * (k + 1)
```

This is closer to the intended Self-MoE-spec interpretation than subtracting a
single delay value, because the hook is applied many times per request.

## Envelope Results for B=2, k=4

| Delay (ms) | `S_max` | `beta_min@1.0x` | `beta_min@1.3x` |
| ---: | ---: | ---: | ---: |
| 0 | 1.013 | 0.994 | NA |
| 0.025 | 1.038 | 0.981 | NA |
| 0.05 | 0.991 | NA | NA |
| 0.1 | 1.024 | 0.988 | NA |
| 0.2 | 1.142 | 0.933 | NA |

## Interpretation

The calibrated runtime delay sweep confirms the local result:

- Small per-collective delays are mostly swallowed by measurement noise and
  scheduling variance in this local DP2 setup.
- Even with `0.2 ms` per delayed collective, `B=2,k=4` reaches only
  `S_max ~= 1.14` and cannot reach 1.3x.
- The result is directionally consistent with the analytical emulation:
  useful speedup requires exposed communication comparable to local compute, not
  tiny per-collective perturbations.

## Caveats

- The p50 values are noisy and not strictly monotonic with delay.
- Each delay value reloads and reruns a new vLLM engine instance.
- The delay is per MoE collective call, and the exact number of calls per
  request is not yet counted.
- This is still single-node DP2+EP, not real multi-node DeepEP/DBO.

## Next Step

Before relying on runtime delay injection quantitatively, add instrumentation to
count how many delayed MoE collectives occur per request or per generated token.
Then calibrate delay values from:

```text
target_step_delay_ms / delayed_collective_count
```

For the research direction itself, the next scientific step remains Phase 03:

```text
measure local-only draft acceptance beta and local expert mass gamma
```

because no timing envelope is useful unless local-only draft acceptance can reach
the required thresholds.
