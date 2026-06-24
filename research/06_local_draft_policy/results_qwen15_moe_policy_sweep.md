# Results: Qwen1.5-MoE Local Draft Policy Sweep

Date: 2026-06-23

## Status

This run evaluates whether simple routing/locality thresholds can isolate a
high-acceptance subset from the Phase 05 one-token acceptance labels.

It uses:

- Phase 03 Qwen1.5-MoE EP2 routing coverage metrics,
- Phase 05 Qwen1.5-MoE EP2 sampled one-token acceptance metrics.

## Generated Artifacts

| Artifact | Purpose |
| --- | --- |
| `evaluate_policy.py` | Joins routing and acceptance metrics, sweeps thresholds |
| `data/qwen15_moe_ep2_policy_joined.csv` | Joined placement/routing/acceptance rows |
| `data/qwen15_moe_ep2_policy_sweep.csv` | Policy threshold sweep |
| `data/qwen15_moe_ep2_policy_sweep.json` | Top policy rows |

## Best Policy Found

The best policies select only one placement/rank/draft configuration:

```text
placement = contiguous_hot_replicated
rank = 1
draft_top_k = 2
sampled_acceptance ~= 0.404
selected_fraction ~= 0.056
```

No policy gets close to the `beta >= 0.8-0.9` target.

## Interpretation

Simple policy gating does not rescue local-only drafting:

- The highest acceptance label available is only about `0.40`.
- Thresholding on `gamma`, top-k overlap, or top-1-local proxy can select that
  best case, but it cannot create a high-acceptance subset.
- The selected fraction is tiny (`1/18` joined rows) and still low-quality.

This suggests the problem is not merely that we are drafting on the wrong tokens.
The local-only draft distribution itself is too different for this checkpoint.

## Conclusion

For Qwen1.5-MoE-A2.7B, local-routing-only policies should stop here:

```text
unconditional local top-k: fails
optimized placement: insufficient
top-1/top-2 local: insufficient
simple confidence/locality gating: insufficient
```

## Recommended Next Direction

The next direction should include a stronger draft mechanism:

1. layer-skip + local routing,
2. learned/lightweight draft head,
3. shared-expert-only draft with explicit acceptance measurement,
4. negative-result framing around local expert coverage limits.
