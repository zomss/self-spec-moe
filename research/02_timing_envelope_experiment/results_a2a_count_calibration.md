# Results: A2A Collective Count Calibration

Date: 2026-06-23

## Goal

Calibrate `VLLM_SELF_SPEC_EMULATE_A2A_DELAY_MS` by measuring how many delayed
MoE EP all-to-all calls occur during the active benchmark window.

## Instrumentation

Added two research-only controls:

```bash
VLLM_SELF_SPEC_LOG_A2A_COUNTS=1
VLLM_SELF_SPEC_A2A_COUNT_ACTIVE_FILE=<path>
```

The all-to-all manager logs:

```text
Self-spec AgRs all2all count: total=<N> active=<M>
```

`active` counts only after the phase-local benchmark helper creates the active
marker file, which happens after engine initialization and warmup.

## Probe

| Item | Value |
| --- | --- |
| GPUs | `CUDA_VISIBLE_DEVICES=6,7` |
| Model config | `Qwen/Qwen1.5-MoE-A2.7B` |
| Weights | dummy |
| Parallelism | DP2+EP |
| Backend | `allgather_reducescatter` |
| Batch | global `B=2` |
| Input/output | `input_len=1`, `output_len=1` |
| Warmup iterations | 1 |
| Benchmark iterations | 3 |

Model config:

| Field | Value |
| --- | ---: |
| `num_hidden_layers` | 24 |
| `num_experts` | 60 |
| `num_experts_per_tok` | 4 |
| `decoder_sparse_step` | 1 |

## Count Result

Each worker logged:

```text
Self-spec AgRs all2all count: total=6336 active=4608
```

For 3 measured iterations, this gives:

```text
active_collectives_per_worker_per_iteration = 4608 / 3 = 1536
```

This is much larger than the naive expectation of:

```text
2 collectives/layer * 24 MoE layers = 48 collectives/token
```

The reason is that the hook currently counts every internal call to
`AgRsAll2AllManager.dispatch_router_logits`, `dispatch`, and `combine`, and the
vLLM execution path performs more internal MoE all-to-all calls per request than
the simple layer-level model suggests. The count is still useful for calibrating
the current hook, but it should not be interpreted as the number of conceptual
EP dispatch/combine phases.

## Delay Calibration for Current Hook

For this current hook and benchmark path:

```text
delay_per_collective =
    target_exposed_delay_per_iteration / 1536
```

Examples:

| Target exposed delay / iteration | Hook delay |
| ---: | ---: |
| 1 ms | 0.00065 ms |
| 2 ms | 0.00130 ms |
| 4 ms | 0.00260 ms |
| 8 ms | 0.00521 ms |

This explains why `0.025-0.2 ms` per hook call was already too coarse: it maps
to tens or hundreds of milliseconds of total injected delay in the current
execution path.

Important: the hook value is **not** the real IB-vs-NVLink latency delta per
conceptual MoE collective. It is the delay per internal hook call. Since this
probe measured `1536` active hook calls per worker per measured iteration,
`0.005 ms` means:

```text
0.005 ms/hook_call * 1536 hook_calls ~= 7.68 ms total injected delay
```

So do not interpret `0.005 ms` as "IB adds only 0.005 ms over NVLink." Interpret
it as "under this hook implementation and benchmark path, 0.005 ms emulates
roughly 8 ms total exposed communication per measured iteration."

The real multi-node question is still:

```text
How many milliseconds of exposed communication does IB/DeepEP add per decode
token after overlap?
```

That requires real profiling or validated traces.

## Next Calibration Sweep

Use smaller hook delays:

```text
0, 0.0005, 0.001, 0.0025, 0.005 ms
```

These correspond approximately to:

```text
0, 0.77, 1.54, 3.84, 7.68 ms
```

of injected delay per measured iteration for this specific DP2+EP path.

## Caveat

The active count is specific to this vLLM execution path, model, backend,
batching, and helper script. Recount before using the delay hook with a
different model/backend/batch path.
