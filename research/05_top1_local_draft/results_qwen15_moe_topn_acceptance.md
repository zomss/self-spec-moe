# Results: Qwen1.5-MoE Top-N Local Draft Acceptance Proxy

Date: 2026-06-23

## Status

This run measures **actual one-token speculative accept/reject samples** from
the draft distribution. It is still not multi-token runtime drafting, but it is
stronger than distribution overlap alone.

The main Self-MoE-spec setting should use the model's default expert top-k and
reroute the same number of experts locally. For Qwen1.5-MoE, that default is
`draft_expert_top_k = 4`. The top-1 and top-2 rows are cheaper-draft variants,
not the main method.

For each prompt:

1. Compute full-router next-token distribution `p`.
2. Compute local-masked draft distribution `q`.
3. Sample draft tokens from `q`.
4. Accept each sampled token with probability:

   ```text
   min(1, p(token) / q(token))
   ```

The expected acceptance equals distribution overlap `sum_i min(p_i, q_i)`, and
the script also reports sampled acceptance.

## Setup

| Item | Value |
| --- | --- |
| Model | `Qwen/Qwen1.5-MoE-A2.7B` |
| Device | GPU 6 |
| dtype | bfloat16 |
| Prompts | 4 short default prompts |
| EP size | 2 |
| Draft routed experts | top-1, top-2, top-4 local |
| Sample count | 512 per prompt |

## Generated Artifacts

| Artifact | Purpose |
| --- | --- |
| `compare_top1_draft_distribution.py` | Actual one-token accept/reject sampler |
| `data/top1_draft_acceptance_qwen15_moe_ep2_prompts4.csv` | Acceptance metrics |
| `data/top1_draft_acceptance_qwen15_moe_ep2_prompts4.json` | Acceptance metrics JSON |
| `logs_top1_draft_acceptance_qwen15_moe_ep2_prompts4.log` | Run log |

## Best Results

| Placement | Rank | Draft top-k | Local experts | Expected acceptance | Sampled acceptance |
| --- | ---: | ---: | ---: | ---: | ---: |
| contiguous_hot_replicated | 1 | 2 | 33 | 0.400 | 0.404 |
| random_hot_replicated | 0 | 2 | 31 | 0.374 | 0.388 |
| random | 0 | 2 | 30 | 0.357 | 0.376 |
| random_hot_replicated | 0 | 4 | 31 | 0.384 | 0.375 |
| random | 0 | 4 | 30 | 0.369 | 0.373 |

## Mean by Draft Top-k

Here `draft_expert_top_k` means the number of local experts used by the draft
MoE layer. It is different from `num_spec_tokens`, the number of speculative
tokens proposed before verification.

| `draft_expert_top_k` | Mean sampled acceptance | Best sampled acceptance |
| ---: | ---: | ---: |
| 1 | 0.131 | 0.192 |
| 2 | 0.269 | 0.404 |
| 4 | 0.244 | 0.375 |

## Interpretation

Neither the same-top-k local draft nor the cheaper top-1/top-2 variants rescue
the method on Qwen1.5-MoE:

- Top-1 local acceptance is very low.
- Top-2 local is better than top-1 and slightly better than top-4 on average.
- Top-4 local is the correct same-top-k local rerouting baseline for this model,
  but it is still far below target acceptance.
- The best sampled acceptance is only about `0.40`.
- This is far below the `beta >= 0.8-0.9` target from the timing envelope.

This means the issue is not only exact top-k locality. Even cheaper top-n local
routing still changes the next-token distribution too much for this checkpoint.

## Conclusion

For Qwen1.5-MoE-A2.7B, local-routing-only drafts should not proceed toward
runtime implementation:

```text
naive local top-k: weak
top-1 local: weak
top-2 local: best of tested variants, but still far too low
hot replication: modest help, insufficient
```

## Recommended Next Direction

The next viable research direction is no longer pure local routing. It should
combine local routing with a separate compute-saving mechanism:

1. layer-skip + local routing,
2. trained/lightweight draft head,
3. shared-expert-only or shared+small-local draft with explicit acceptance
   measurement,
4. or negative-result framing for local-only MoE speculation.
