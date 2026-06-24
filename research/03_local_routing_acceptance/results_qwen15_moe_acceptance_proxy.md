# Results: Qwen1.5-MoE Local-Masked Router Acceptance Proxy

Date: 2026-06-23

## Status

This is a direct next-token distribution proxy for local-only routing. It is
still not full speculative acceptance over multi-token drafts, but it is much
closer to the acceptance question than router coverage alone.

## Setup

| Item | Value |
| --- | --- |
| Model | `Qwen/Qwen1.5-MoE-A2.7B` |
| Device | GPU 6 |
| dtype | bfloat16 |
| Prompts | 4 short default prompts |
| Max length | 32 |
| EP size | 2 |
| Placement policies | contiguous, random, contiguous + hot replication |

The script compares:

```text
full-router next-token distribution
vs.
local-masked-router next-token distribution
```

The primary proxy metric is distribution overlap:

```text
sum_i min(p_full(i), p_local(i))
```

This is an upper-bound-like acceptance proxy: values near 1 mean very similar
next-token distributions; low values imply poor speculative acceptance.

## Generated Artifacts

| Artifact | Purpose |
| --- | --- |
| `compare_masked_router_distribution.py` | Runs full-vs-local next-token proxy |
| `data/masked_router_proxy_qwen15_moe_ep2_prompts4.csv` | Proxy metrics |
| `data/masked_router_proxy_qwen15_moe_ep2_prompts4.json` | Proxy metrics JSON |
| `logs/masked_router_proxy_qwen15_moe_ep2_prompts4.log` | Run log |

## Results

| Placement | Rank | Local experts | Distribution overlap | KL(full || local) | Top-1 match |
| --- | ---: | ---: | ---: | ---: | ---: |
| contiguous | 0 | 30 | 0.087 | 5.148 | 0.25 |
| contiguous | 1 | 30 | 0.305 | 2.488 | 0.25 |
| random | 0 | 30 | 0.369 | 5.596 | 0.25 |
| random | 1 | 30 | 0.052 | 6.192 | 0.25 |
| contiguous_hot_replicated | 0 | 35 | 0.206 | 7.025 | 0.25 |
| contiguous_hot_replicated | 1 | 33 | 0.347 | 4.677 | 0.00 |

## Interpretation

The acceptance proxy is poor for Qwen1.5-MoE EP2:

- Best observed distribution overlap is only `~0.37`.
- KL divergence is large.
- Top-1 match is low and unstable.
- Hot replication does not reliably improve the next-token distribution.

This strongly suggests that naive local-masked routing will not reach the
`beta >= 0.8-0.9` target needed by the timing envelope.

## Conclusion

For Qwen1.5-MoE-A2.7B, the naive local-top-k draft direction should not proceed
to runtime implementation.

The most reasonable next research direction is a pivot:

1. shared/hot expert plus top-1 local draft,
2. layer-skip + local-routing hybrid,
3. training-free but cheaper draft that does not require exact local top-k
   agreement,
4. or use this result as evidence that local expert coverage is a core
   limitation.
