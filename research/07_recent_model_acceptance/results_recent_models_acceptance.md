# Results: Local-Draft Acceptance on Recent MoE Models

Date: 2026-06-24

## Status

One-token speculative accept/reject proxy (same method as Phase 05) on two recent
MoE checkpoints. Verification is exact full routing; only the draft routing is
masked to device-local experts. This is still a one-token proxy, so real
multi-token `beta` would be lower; these numbers are an optimistic upper bound.

Both models have **no shared expert**, so this set tests recency and expert
granularity, not the shared-expert anchor hypothesis.

## Setup

| Item | Value |
| --- | --- |
| Models | `Qwen/Qwen3-30B-A3B`, `openai/gpt-oss-20b` |
| Hardware | 1x H100 80GB each |
| dtype | bfloat16 (GPT-OSS MXFP4 dequantized to bf16) |
| Prompts | 16, domain-bucketed (general / chat / code / math) |
| EP sizes | 2, 4, 8 |
| Placements | contiguous, random, +hot-replicated (8 hot experts) |
| Samples | 512 per (prompt, config) |

## Results (main method: `draft_top_k = default top-k`)

### Qwen3-30B-A3B (128 experts, top-8)

| EP size | Best sampled acceptance | gamma | top-1 (greedy) match |
| ---: | ---: | ---: | ---: |
| 2 | 0.504 | 0.53 | 0.56 |
| 4 | 0.309 | 0.34 | 0.13 |
| 8 | 0.119 | 0.23 | 0.06 |

### GPT-OSS-20B (32 experts, top-4)

| EP size | Best sampled acceptance | gamma | top-1 (greedy) match |
| ---: | ---: | ---: | ---: |
| 2 | 0.696 | 0.67 | 0.38 |
| 4 | 0.584 | 0.52 | 0.38 |
| 8 | 0.518 | 0.40 | 0.44 |

Best configs are hot-replicated contiguous/random placements at `draft_top_k =
default` (confirming the same-top-k local rerouting is the right main method;
cheaper top-1/top-2 variants were not better).

## Comparison to prior phases

| Model | Released | Experts/top-k | Shared | Best sampled (EP2) | Best sampled (EP8) |
| --- | --- | --- | --- | ---: | ---: |
| PowerMoE-3B | 2024 | 32/? | no | weak (~0.4-class) | - |
| Qwen1.5-MoE-A2.7B | 2024 | 60/4 | small | ~0.40 | <0.1-class |
| Qwen3-30B-A3B | 2025 | 128/8 | no | 0.50 | 0.12 |
| GPT-OSS-20B | 2025 | 32/4 | no | 0.70 | 0.52 |

Recent models are somewhat better than Qwen1.5-MoE at low EP. GPT-OSS-20B at EP2
(0.70) is the best result in the whole project so far. But none clears the
`beta >= 0.8` bar, and the gain is concentrated at low EP.

## Key finding: the two requirements are anti-correlated

Acceptance is highest at **EP2** (single-node-like, where exposed all-to-all is
small, `f ~= 0.15`, so the communication ceiling is only ~1.05-1.14x) and
collapses at **EP8** (multi-node, large `f`, the only regime with a real
communication advantage):

```text
acceptance is best exactly where the communication advantage is smallest,
and collapses exactly where the communication advantage is largest.
```

This is a structural obstacle, not a model artifact: more EP shards -> fewer
routed experts per shard -> lower local gate mass `gamma` -> lower acceptance.
GPT-OSS degrades more gracefully than Qwen3 only because it is coarser (32 vs 128
experts), so each shard keeps more mass; but coarse models are also the ones
least likely to be served at high EP.

## Caveats

- **Greedy vs sampled.** GPT-OSS reaches 0.70 sampled (temperature-1) but only
  0.38 top-1 match, so under greedy/low-temperature decode (the latency-critical
  case) its effective acceptance is ~0.38. Qwen3's top-1 (0.56 at EP2) exceeds
  its sampled, so it is less temperature-sensitive but lower overall.
- **One-token proxy.** Multi-token drafting would compound divergence, lowering
  real `beta` below these numbers.
- **No shared expert tested.** The 00_proposal anchor hypothesis (a large
  always-local shared expert provides an EP-invariant acceptance floor) is still
  untested. The shared expert mass does not shrink with EP, so it is the one
  mechanism that could break the anti-correlation above.

## Conclusion

Across four checkpoints (old/recent, fine/coarse, all without a shared anchor),
local-only drafting does not reach the acceptance bar, and is weakest in the
high-EP multi-node target regime. The negative result is now cross-architecture
and recent.

## Recommended next step

The single most decisive remaining experiment is to test a **shared-expert** MoE
and check whether the always-local shared expert provides an EP-invariant
acceptance floor that survives high EP. Recent runnable candidates on this node:

```text
GLM-4.5-Air, Llama-4-Scout, or DeepSeek-V3/V2 (shared-expert architectures)
```

If a shared-expert model still collapses at high EP, the negative result is
complete and should be written up. If its acceptance holds up at high EP, the
method is alive specifically for shared-expert architectures.
