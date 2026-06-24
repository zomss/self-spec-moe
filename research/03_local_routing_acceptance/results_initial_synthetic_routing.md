# Results: Initial Synthetic Local-Routing Coverage

Date: 2026-06-23

## Status

This is a lightweight Phase 03 validation run. It does **not** measure real
draft-token acceptance `beta` yet.

It validates the local-routing metrics pipeline on synthetic router logits with
the shape of `Qwen/Qwen1.5-MoE-A2.7B`:

| Field | Value |
| --- | ---: |
| MoE layers | 24 |
| Experts | 60 |
| top-k | 4 |
| EP size | 2 |
| Tokens | 512 synthetic tokens |

## Generated Artifacts

| Artifact | Purpose |
| --- | --- |
| `simulate_local_routing.py` | Computes `gamma`, top-k overlap, and full-top-k-local rate |
| `data/local_routing_metrics_synthetic_uniform_qwen15_moe_shape.csv` | Uniform synthetic router logits |
| `data/local_routing_metrics_synthetic_qwen15_moe_shape.csv` | Hot-expert-biased synthetic router logits |

## Metrics

| Metric | Meaning |
| --- | --- |
| `gamma` | Local expert probability mass |
| top-k overlap | Fraction of true top-k experts available locally |
| full-top-k-local rate | Fraction of tokens where all true top-k experts are local |

Rows below use the `best_gamma_rank` upper-bound policy, which assigns each
token to the rank with highest local expert mass.

## Uniform Synthetic Router

| Placement | `gamma` | top-k overlap | full-top-k-local rate |
| --- | ---: | ---: | ---: |
| contiguous | 0.563 | 0.651 | 0.112 |
| round_robin | 0.563 | 0.652 | 0.113 |
| random | 0.563 | 0.652 | 0.115 |
| hot_grouped | 0.563 | 0.651 | 0.112 |
| contiguous_hot_replicated | 0.563 | 0.651 | 0.112 |

Interpretation: without expert locality or hot experts, local-only routing is
weak. Even with the best local rank chosen per token, all top-k experts are local
only about 11% of the time.

## Hot-Expert-Biased Synthetic Router

This synthetic run adds a bias toward 8 hot experts. It is not a model result;
it only checks how the metrics respond when routing locality exists.

| Placement | `gamma` | top-k overlap | full-top-k-local rate |
| --- | ---: | ---: | ---: |
| contiguous | 0.720 | 0.897 | 0.640 |
| round_robin | 0.585 | 0.615 | 0.054 |
| random | 0.669 | 0.794 | 0.358 |
| hot_grouped | 0.720 | 0.897 | 0.640 |
| contiguous_hot_replicated | 0.803 | 0.954 | 0.824 |

Interpretation: if routing mass is concentrated in local or replicated hot
experts, the local-only draft becomes much more plausible. Hot replication gives
the strongest synthetic coverage, but it increases the average local expert set
size from 30 to 34 in this setup.

## Phase 03 Implication

The key question is now empirical:

```text
Do real MoE router logits look closer to the uniform case or the hot/local case?
```

If real routing behaves like the uniform synthetic run, Self-MoE-spec is unlikely
to reach the required acceptance. If real routing has strong locality or hot
expert concentration, the idea remains plausible.

## Next Step

Run the same metrics on real router logits from an actual MoE checkpoint. The
minimum next metric is:

```text
gamma and top-k overlap from real router logits
```

Only after real routing coverage looks promising should we implement full
local-only draft-token generation and measure acceptance `beta`.
