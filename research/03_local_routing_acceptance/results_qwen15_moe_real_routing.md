# Results: Real Router Coverage on Qwen1.5-MoE-A2.7B

Date: 2026-06-23

## Status

This is the second Phase 03 run on real router logits, and the first one on a
Qwen-style MoE checkpoint.

It measures routing coverage metrics only:

- `gamma`,
- top-k overlap,
- full-top-k-local rate.

It does **not** yet measure draft-token acceptance `beta`.

## Model and Data

| Item | Value |
| --- | --- |
| Model | `Qwen/Qwen1.5-MoE-A2.7B` |
| Device | GPU 6 |
| dtype | bfloat16 |
| Prompts | 4 short default prompts |
| Max length | 32 |
| Extracted shape | `[24 layers, 40 tokens, 60 experts]` |
| top-k | 4 |
| EP sizes | 2, 4, 8 |

Generated artifacts:

| Artifact | Purpose |
| --- | --- |
| `data/router_logits_qwen15_moe_prompts4_len32.npz` | Extracted real router logits |
| `data/router_logits_qwen15_moe_prompts4_len32.summary.json` | Extraction summary |
| `data/local_routing_metrics_qwen15_moe_prompts4_len32_ep*.csv` | Routing coverage metrics |
| `logs/extract_qwen15_moe_prompts4_len32.log` | Extraction log |

## Best-Rank Upper-Bound Results

Rows below use `best_gamma_rank`, which chooses the rank with highest local
expert mass per token. This is optimistic.

### EP Size 2

| Placement | `gamma` | top-k overlap | full-top-k-local rate | Avg local experts |
| --- | ---: | ---: | ---: | ---: |
| contiguous | 0.612 | 0.648 | 0.127 | 30 |
| round_robin | 0.615 | 0.654 | 0.119 | 30 |
| random | 0.618 | 0.665 | 0.132 | 30 |
| hot_grouped | 0.612 | 0.639 | 0.099 | 30 |
| contiguous_hot_replicated | 0.672 | 0.698 | 0.200 | 34 |

### EP Size 4

| Placement | `gamma` | top-k overlap | full-top-k-local rate | Avg local experts |
| --- | ---: | ---: | ---: | ---: |
| contiguous | 0.416 | 0.459 | 0.014 | 15 |
| round_robin | 0.415 | 0.454 | 0.005 | 15 |
| random | 0.417 | 0.456 | 0.013 | 15 |
| hot_grouped | 0.417 | 0.462 | 0.008 | 15 |
| contiguous_hot_replicated | 0.510 | 0.552 | 0.068 | 21 |

### EP Size 8

| Placement | `gamma` | top-k overlap | full-top-k-local rate | Avg local experts |
| --- | ---: | ---: | ---: | ---: |
| contiguous | 0.318 | 0.351 | 0.004 | 7.5 |
| round_robin | 0.315 | 0.343 | 0.001 | 7.5 |
| random | 0.314 | 0.344 | 0.001 | 7.5 |
| hot_grouped | 0.319 | 0.353 | 0.002 | 7.5 |
| contiguous_hot_replicated | 0.425 | 0.468 | 0.031 | 14.5 |

## Interpretation

The Qwen1.5-MoE real-router result is also weak for local-only drafting under
simple placement policies.

Key observations:

- EP2 coverage is moderate but probably insufficient: `gamma ~= 0.61-0.62` and
  full top-k local rate only `~10-13%` without replication.
- EP4 and EP8 degrade sharply, as expected when each rank holds fewer experts.
- Replicating 8 hot experts helps, but not enough for the target acceptance
  range. At EP8, hot replication gives only `gamma ~= 0.43` and full top-k
  local rate `~3%`.
- Placement policies are very similar, which suggests no strong simple
  expert-locality signal in this small prompt sample.

## Phase 03 Implication

This result makes the original local-only top-k MoE draft risky. If actual
acceptance correlates with these coverage metrics, `beta >= 0.8-0.9` is unlikely
for simple local-only routing on this checkpoint.

The most promising pivots are:

1. **Make the draft cheaper**, e.g. shared expert plus top-1 local expert.
2. **Use stronger replication**, especially shared/hot experts, if memory allows.
3. **Use locality-aware request placement**, but only if larger prompt/domain
   traces reveal stronger structure than this sample.
4. **Hybridize with layer skipping**, so the draft saves compute as well as
   communication.

## Next Step

Run a direct acceptance proxy instead of only coverage:

```text
compare full-router next-token logits vs local-masked-router next-token logits
```

for EP2 first. If the next-token distributions diverge strongly, do not proceed
to runtime local drafting for this checkpoint.
