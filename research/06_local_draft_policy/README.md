# Phase 06: Local Draft Policy

Source phase: [`../05_top1_local_draft`](../05_top1_local_draft)

## Objective

Design and evaluate a **policy** for when and how to use local MoE drafting.

The current evidence says unconditional local routing is too weak. Phase 06
therefore asks:

```text
Can we selectively draft only when router/locality signals predict high
acceptance?
```

## Motivation

Previous phases showed:

- naive local top-k has poor coverage,
- optimized placement helps only modestly,
- top-1/top-2 local drafts still have low one-token sampled acceptance,
- timing can help only when acceptance is high.

Therefore the next contribution should be a policy, not just a routing mask.

## Policy Inputs

| Signal | Meaning |
| --- | --- |
| `gamma` | Local expert probability mass |
| top-1 local | Whether the strongest routed expert is local |
| router margin | Confidence gap between best and next-best experts |
| router entropy | How concentrated the router distribution is |
| hot/shared coverage | Whether replicated hot/shared experts cover most mass |
| routing stability | Whether recent tokens route similarly |
| placement id | Which rank/expert placement is being used |

## Policy Output

The policy should choose:

| Output | Meaning |
| --- | --- |
| draft enabled | Whether to draft locally or fall back to full MoE |
| draft expert set | top-1 local, top-2 local, hot/shared + local, etc. |
| draft length `k` | `0, 1, 2, 4, 8` |
| target rank/group | Which local placement/rank should handle the request |

## Baseline Policy Sketch

```text
if gamma >= gamma_threshold
and top1 expert is local
and router entropy <= entropy_threshold:
    if margin is high:
        k = 4
    else:
        k = 1 or 2
else:
    k = 0  # full MoE decode
```

This reframes Self-MoE-spec from unconditional local drafting to opportunistic
local drafting.

## Target Outputs

| Output | Path | Purpose |
| --- | --- | --- |
| Policy evaluator | `evaluate_policy.py` | Evaluate gating policies on Phase 03/04 traces |
| Policy metrics | `data/policy_metrics_*.csv` | Coverage, predicted acceptance, and enabled-token rate |
| Summary | `results_*.md` | Whether a policy can isolate high-acceptance cases |

## Go/No-Go Criteria

Proceed only if a policy can find a non-trivial subset of tokens/requests with:

- predicted or measured acceptance near the required `beta`,
- enough enabled tokens to matter for throughput/latency,
- no dependence on unrealistic expert replication.

If the policy only enables a tiny fraction of tokens, the research should pivot
to negative-result framing or to a learned/trained draft.

## Expected First Artifact

```text
research/06_local_draft_policy/evaluate_policy.py
```

Start with trace-level policy evaluation using existing `gamma`, top-k overlap,
entropy, and margin metrics. Add actual acceptance labels when available.

## Current Artifact

`evaluate_policy.py` joins Phase 03/05 routing and acceptance metrics and sweeps
simple thresholds over `gamma`, top-1 locality, and top-k overlap.

## Current Result

`results_qwen15_moe_policy_sweep.md` shows that simple policy gating cannot
isolate a high-acceptance subset for Qwen1.5-MoE. The best selected case remains
only about `0.40` sampled acceptance.
