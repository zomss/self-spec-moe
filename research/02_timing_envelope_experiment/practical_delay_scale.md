# Practical Scale for Per-Collective Delay

Date: 2026-06-23

## Question

What is a practical value for `VLLM_SELF_SPEC_EMULATE_A2A_DELAY_MS`?

## Short Answer

The hook is **per MoE collective**, so practical values are likely in the
**tens to hundreds of microseconds**:

```text
0.025 ms, 0.05 ms, 0.1 ms, 0.2 ms
```

Values like `1 ms` or `4 ms` per collective are usually too large unless the
system is extremely communication-bound or badly configured.

## Why Small Per-Collective Delays Matter

MoE EP usually performs two communication phases per MoE layer:

```text
dispatch + combine
```

If a model has many MoE layers, per-collective delay accumulates quickly:

```text
exposed_comm_per_token ~= delay_per_collective * collectives_per_token
collectives_per_token ~= 2 * num_moe_layers
```

For example, if there are 30 MoE layers:

| Delay per collective | Approx exposed communication per token |
| ---: | ---: |
| 0.025 ms | 1.5 ms |
| 0.05 ms | 3.0 ms |
| 0.1 ms | 6.0 ms |
| 0.2 ms | 12.0 ms |

This means `0.1 ms` per collective can already emulate a very large per-token
multi-node communication penalty.

## IB / Collective Scale

Rough practical scale for modern H100-class multi-node IB systems:

| Component | Plausible scale |
| --- | ---: |
| Raw NIC / IB one-way latency | 1-5 us |
| NCCL collective launch and synchronization | 10-50 us |
| Small exposed all-to-all / allgather latency | 20-200 us |
| Larger MoE dispatch/combine under payload pressure | 0.2-1+ ms |
| Total per-token exposed communication across layers | Several ms if repeated across many MoE layers |

The relevant quantity for this hook is the **exposed per-collective** latency,
not raw NIC latency and not total per-token communication.

## Calibration Rule

Do not choose the hook value directly from desired per-token delay. First count
how many delayed collectives happen per generated token:

```text
delay_per_collective_ms =
    target_exposed_comm_per_token_ms / delayed_collectives_per_token
```

Example:

```text
target_exposed_comm_per_token = 4 ms
delayed_collectives_per_token = 60
delay_per_collective = 4 / 60 = 0.0667 ms
```

So a hook value near `0.05-0.1 ms` may already represent a realistic high
multi-node communication regime.

## Recommended Sweep Values

Use:

```text
0, 0.025, 0.05, 0.1, 0.2 ms
```

Then, after counting delayed collectives per token, add targeted values that map
to:

```text
target exposed communication per token = 2, 4, 8 ms
```

## Interpretation of Current Sweep

The calibrated GPU6/7 sweep showed that even `0.2 ms` per collective did not
reach a strong envelope locally. Since `0.2 ms` per collective can represent a
large total per-token communication penalty, the local runtime sweep should not
be overinterpreted as a realistic multi-node proof.

The next necessary calibration is to count delayed collective calls per token.

## Updated Hook-Specific Calibration

The first count probe for the current hook measured:

```text
active hook calls per worker per measured iteration = 1536
```

Therefore, for the current implementation, practical hook values are much
smaller than the conceptual per-layer collective values above:

| Target injected delay / measured iteration | Hook value |
| ---: | ---: |
| 1 ms | 0.00065 ms |
| 2 ms | 0.00130 ms |
| 4 ms | 0.00260 ms |
| 8 ms | 0.00521 ms |

For the next runtime sweep, use:

```text
0, 0.0005, 0.001, 0.0025, 0.005 ms
```

See `results_a2a_count_calibration.md`.

Do not interpret these hook values as real IB-vs-NVLink latency. They are
implementation-specific values that multiply by the current hook-call count.
For example, `0.005 ms` maps to about `7.68 ms` total injected delay in the
current DP2+EP benchmark path.
