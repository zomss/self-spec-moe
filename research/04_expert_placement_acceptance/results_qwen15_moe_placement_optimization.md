# Results: Qwen1.5-MoE Expert Placement Optimization

Date: 2026-06-23

## Status

This is the first Phase 04 placement-optimization run.

It consumes real Qwen1.5-MoE router logits from Phase 03 and produces expert
placement maps plus coverage metrics. It still does **not** measure actual
accepted draft tokens.

## Input

| Item | Value |
| --- | --- |
| Router logits | `../03_local_routing_acceptance/data/router_logits_qwen15_moe_prompts4_len32.npz` |
| Shape | `[24 layers, 40 tokens, 60 experts]` |
| top-k | 4 |
| EP sizes | 2, 4, 8 |
| Hot experts replicated | 8 |

## Placement Policies

Generated:

- contiguous,
- random,
- load-balanced by expert probability mass,
- coactivation-greedy,
- hot-replicated variants of each.

Artifacts:

| Artifact | Purpose |
| --- | --- |
| `optimize_expert_placement.py` | Placement optimizer and coverage evaluator |
| `data/qwen15_moe_prompts4_len32_maps.json` | Expert-to-rank maps |
| `data/qwen15_moe_prompts4_len32_coverage_metrics.csv` | Coverage metrics |

## Best Results by EP Size

### EP Size 2

| Placement | Replicated | `gamma` | top-k overlap | top-1 local | full top-k local |
| --- | --- | ---: | ---: | ---: | ---: |
| random_hot_replicated | yes | 0.684 | 0.723 | 0.901 | 0.210 |
| coactivation_greedy_hot_replicated | yes | 0.679 | 0.709 | 0.863 | 0.209 |
| load_balanced_hot_replicated | yes | 0.674 | 0.706 | 0.901 | 0.196 |
| random | no | 0.623 | 0.662 | 0.854 | 0.123 |

### EP Size 4

| Placement | Replicated | `gamma` | top-k overlap | top-1 local | full top-k local |
| --- | --- | ---: | ---: | ---: | ---: |
| random_hot_replicated | yes | 0.516 | 0.566 | 0.855 | 0.065 |
| coactivation_greedy_hot_replicated | yes | 0.511 | 0.567 | 0.841 | 0.070 |
| load_balanced_hot_replicated | yes | 0.510 | 0.562 | 0.858 | 0.061 |
| random | no | 0.422 | 0.471 | 0.814 | 0.007 |

### EP Size 8

| Placement | Replicated | `gamma` | top-k overlap | top-1 local | full top-k local |
| --- | --- | ---: | ---: | ---: | ---: |
| coactivation_greedy_hot_replicated | yes | 0.431 | 0.490 | 0.858 | 0.052 |
| random_hot_replicated | yes | 0.427 | 0.474 | 0.848 | 0.042 |
| load_balanced_hot_replicated | yes | 0.425 | 0.470 | 0.865 | 0.031 |
| coactivation_greedy | no | 0.320 | 0.366 | 0.814 | 0.007 |

## Interpretation

Optimized placement improves coverage slightly, especially with hot replication,
but it does not change the overall conclusion:

- `gamma` remains far below the synthetic hot-local case.
- Full top-k local rate remains low.
- EP4 and EP8 are especially weak.
- Top-1 local rate is surprisingly high, which supports trying a **top-1 local
  draft** variant, but not naive local top-k drafting.

The best EP2 replicated placement reaches:

```text
gamma ~= 0.68
top-k overlap ~= 0.72
full top-k local ~= 0.21
```

This still looks unlikely to produce `beta >= 0.8-0.9` for local top-k draft.

## Recommendation

Do not spend more time on naive local top-k placement optimization for this
checkpoint. The more promising next experiment is:

```text
shared/hot expert + top-1 local expert draft
```

because top-1 local rates are high even when full top-k locality is poor.

Phase 05 has been initialized at
[`../05_top1_local_draft`](../05_top1_local_draft) for this pivot.
