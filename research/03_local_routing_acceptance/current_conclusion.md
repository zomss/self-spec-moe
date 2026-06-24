# Current Phase 03 Conclusion

Date: 2026-06-23

## Summary

Real-router coverage is weak for both tested MoE checkpoints under simple local
expert placement policies.

| Model | EP size | Best simple `gamma` | Best simple top-k overlap | Full top-k local rate |
| --- | ---: | ---: | ---: | ---: |
| PowerMoE-3B | 2 | ~0.59 | ~0.60 | ~0.5% |
| Qwen1.5-MoE-A2.7B | 2 | ~0.62 | ~0.66 | ~13% |
| Qwen1.5-MoE-A2.7B | 4 | ~0.42 | ~0.46 | ~1% |
| Qwen1.5-MoE-A2.7B | 8 | ~0.32 | ~0.35 | <1% |

Hot expert replication helps but does not reach the likely acceptance target:

| Model | EP size | Replicated `gamma` | Replicated top-k overlap | Full top-k local rate |
| --- | ---: | ---: | ---: | ---: |
| PowerMoE-3B | 2 | ~0.68 | ~0.70 | ~3% |
| Qwen1.5-MoE-A2.7B | 2 | ~0.67 | ~0.70 | ~20% |
| Qwen1.5-MoE-A2.7B | 8 | ~0.43 | ~0.47 | ~3% |

## Interpretation

The original idea of drafting with only local top-k experts is now risky. The
coverage metrics are far below the synthetic hot-local case and likely too low
to reach `beta >= 0.8-0.9`.

This does not fully disprove Self-MoE-spec. We have not yet measured actual
speculative acceptance, and we have not tuned expert placement beyond simple
policies. The result says simple placements are weak, not that optimized
placement cannot work.

## Recommended Next Step

Before pivoting, tune expert placement from router traces and measure actual
acceptance:

1. optimize expert-to-rank maps,
2. generate local-masked draft tokens,
3. verify with the full model,
4. measure `beta` for `k = 1, 2, 4, 8`.

If optimized placement still fails, then prioritize variants that make the draft
cheaper or less dependent on exact top-k local overlap:

- shared expert + top-1 local expert,
- hot/shared expert replication,
- locality-aware request-to-rank placement,
- layer-skip plus local-routing hybrid.

## Next Experiment

Before implementing runtime local drafting, run a direct acceptance proxy:

```text
full-router next-token distribution
vs.
local-masked-router next-token distribution
```

Start with Qwen1.5-MoE EP2 because it has the strongest current real-router
coverage among tested settings.

This proxy has now been run in `results_qwen15_moe_acceptance_proxy.md`. The
best observed next-token distribution overlap is only about `0.37`, but this is
not actual acceptance. Phase 04 should test optimized placement and exact
verification before making the final go/no-go decision.
