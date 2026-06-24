# Phase 05: Cheaper Local Draft Variant

Source phase: [`../04_expert_placement_acceptance`](../04_expert_placement_acceptance)

## Objective

Evaluate cheaper local draft variants after Phase 04 showed that same-top-k
local drafting is weak but top-1 locality is high.

The phase asks:

```text
Can a shared/hot expert + top-1 local expert draft preserve enough quality while
reducing draft compute enough to beat the timing envelope?
```

This is a **variant**, not the main Self-MoE-spec method. The main method keeps
the model's default expert top-k and reroutes that many experts into the local
expert set.

## Motivation

Phase 04 found:

```text
top-1 local rate ~= 0.85-0.90
full top-k local rate ~= 0.21 at best
```

So exact local top-k agreement is too hard, but top-1 locality may be enough for
a cheaper approximate draft.

## Candidate Drafts

| Draft variant | Meaning |
| --- | --- |
| top-1 local | Use only the strongest local routed expert. |
| hot/shared + top-1 local | Always include hot/shared experts plus the strongest local expert. |
| top-2 local | Intermediate comparison between top-1 and full local top-k. |
| layer-skip + top-1 local | Optional hybrid if top-1 local alone is insufficient. |

Terminology:

```text
draft_expert_top_k = number of experts used by the draft MoE layer
num_spec_tokens = number of speculative draft tokens before verification
```

Avoid using `k` alone because it can mean either expert top-k or draft token
length.

For the main Self-MoE-spec method:

```text
draft_expert_top_k = default model expert_top_k
```

For Qwen1.5-MoE-A2.7B, the default is `draft_expert_top_k = 4`.

## Metrics

| Metric | Purpose |
| --- | --- |
| distribution overlap | Proxy for speculative acceptance |
| top-1 token match | Greedy agreement with full model |
| KL(full || draft) | Distribution divergence |
| estimated draft compute ratio `c` | Needed for timing envelope |
| speedup envelope | Plug `c` and measured/proxy acceptance into Phase 01/02 formulas |

## Initial Setting

| Dimension | Value |
| --- | --- |
| Model | `Qwen/Qwen1.5-MoE-A2.7B` |
| EP size | 2 first |
| Placement | best Phase 04 placement candidates |
| Prompts | same short default prompts first |
| Draft depth | start with one-token proxy, then `k = 1, 2, 4` if promising |

## Go/No-Go Criteria

Proceed only if:

- distribution overlap improves substantially over naive local masking,
- top-1 token match improves,
- estimated draft compute ratio is meaningfully below full local top-k,
- speedup envelope improves under plausible communication exposure.

Stop local-routing-only drafting if top-1 local still has poor overlap.

## Expected First Artifact

```text
research/05_top1_local_draft/compare_top1_draft_distribution.py
```

This should extend the Phase 03 distribution proxy to test top-1 local, top-2
local, and hot/shared + top-1 local draft variants.

## Current Artifact

`compare_top1_draft_distribution.py` samples actual one-token speculative
accept/reject decisions from the draft distribution and compares top-1/top-2/top-4
local variants.

## Current Result

`results_qwen15_moe_topn_acceptance.md` shows that local drafts with
`draft_expert_top_k = 1, 2, 4` still have low actual sampled one-token
acceptance on Qwen1.5-MoE. `draft_expert_top_k = 2` is best but only reaches
about `0.40`, far below the `0.8-0.9` target.

Phase 06 has been initialized at
[`../06_local_draft_policy`](../06_local_draft_policy) to test selective
drafting policies instead of unconditional local routing.
