# Results: Real Router Coverage on PowerMoE-3B

Date: 2026-06-23

## Status

This is the first Phase 03 run on real router logits.

It measures routing coverage metrics only:

- `gamma`,
- top-k overlap,
- full-top-k-local rate.

It does **not** yet measure draft-token acceptance `beta`.

## Model and Data

| Item | Value |
| --- | --- |
| Model | `ibm-research/PowerMoE-3b` |
| Device | GPU 6 |
| dtype | bfloat16 |
| Prompts | 4 short default prompts |
| Max length | 32 |
| Extracted shape | `[32 layers, 40 tokens, 40 experts]` |
| top-k | 8 |
| EP size | 2 |

Generated artifacts:

| Artifact | Purpose |
| --- | --- |
| `data/router_logits_powermoe3b_prompts4_len32.npz` | Extracted real router logits |
| `data/router_logits_powermoe3b_prompts4_len32.summary.json` | Extraction summary |
| `data/local_routing_metrics_powermoe3b_prompts4_len32.csv` | Routing coverage metrics |
| `logs/extract_powermoe3b_prompts4_len32.log` | Extraction log |

## Best-Rank Upper-Bound Results

Rows below use `best_gamma_rank`, which chooses the rank with highest local
expert mass per token. This is optimistic; a real scheduler may not always pick
that rank.

| Placement | `gamma` | top-k overlap | full-top-k-local rate | Avg local experts |
| --- | ---: | ---: | ---: | ---: |
| contiguous | 0.589 | 0.601 | 0.005 | 20 |
| round_robin | 0.586 | 0.601 | 0.002 | 20 |
| random | 0.579 | 0.586 | 0.004 | 20 |
| hot_grouped | 0.581 | 0.597 | 0.005 | 20 |
| contiguous_hot_replicated | 0.682 | 0.698 | 0.030 | 24 |

## Interpretation

This real-router result is much closer to the weak/uniform synthetic case than
to the hot-local synthetic case.

Key observations:

- Best-rank `gamma` is only about `0.58-0.59` for non-replicated placements.
- Average top-k overlap is only about `0.59-0.60`.
- Full top-k local rate is almost zero (`0.2-0.5%`) without replication.
- Replicating 8 hot experts helps, but only reaches `gamma ~= 0.68` and full
  top-k local rate `~3%`.

This suggests local-only routing is unlikely to produce high speculative
acceptance for this model under simple placement policies.

## Caveats

- This is PowerMoE-3B, not the target DeepSeek/Qwen large MoE.
- Only 4 short prompts were used.
- This measures routing coverage, not actual token acceptance.
- `best_gamma_rank` is optimistic.

## Phase 03 Implication

For this real model/checkpoint, the local-routing signal is weak. The next real
model to test should be a DeepSeek/Qwen-style MoE with known shared experts or
stronger hot-expert locality.

If larger target models show similar coverage, Self-MoE-spec should pivot toward:

- cheaper drafts,
- stronger expert replication,
- locality-aware request placement,
- or a hybrid with layer skipping.
