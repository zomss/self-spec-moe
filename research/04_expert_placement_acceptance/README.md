# Phase 04: Expert Placement and Actual Acceptance

Source phase: [`../03_local_routing_acceptance`](../03_local_routing_acceptance)

## Objective

Move beyond coverage and next-token distribution proxies by tuning expert
placement and measuring actual speculative acceptance.

The phase asks:

```text
Can optimized expert placement make local-only drafts accepted by exact full-MoE
verification often enough to be useful?
```

## Motivation

Phase 03 measured:

- `gamma`,
- top-k overlap,
- full-top-k-local rate,
- next-token distribution overlap.

Those are useful proxies, but they are not actual speculative acceptance. The
next step is to optimize placement from router traces and then directly measure
accepted draft tokens.

## Target Outputs

| Output | Path | Purpose |
| --- | --- | --- |
| Placement optimizer | `optimize_expert_placement.py` | Build expert-to-rank maps from router traces |
| Placement maps | `data/placement_*.json` | Expert placement for each policy |
| Acceptance runner | `measure_acceptance.py` | Generate local drafts and verify with full model |
| Metrics CSV | `data/acceptance_metrics_*.csv` | Actual `beta` by placement and draft depth |
| Summary | `results_*.md` | Go/no-go interpretation |

## Placement Policies

Start with:

1. contiguous EP-style placement,
2. random placement,
3. hot-expert replication,
4. co-activation-aware placement,
5. domain/prompt-cluster-aware placement if enough traces exist.

## Acceptance Metrics

| Metric | Meaning |
| --- | --- |
| `beta@1` | Acceptance at first draft token |
| `beta@2`, `beta@4`, `beta@8` | Acceptance decay by draft position |
| expected accepted tokens | `E = 1 + beta + beta^2 + ...` or measured cycle acceptance |
| speedup envelope | Timing-envelope speedup after plugging measured acceptance |

## Go/No-Go Criteria

Proceed to runtime local drafting only if optimized placement gives:

- `beta >= 0.8` for high exposed-communication regimes, or
- `beta >= 0.9` for moderate exposed communication / DBO-composed regimes, and
- acceptance does not collapse by draft positions `2-4`.

If optimized placement still gives low acceptance, pivot to:

- shared/hot expert + top-1 local draft,
- layer-skip + local-routing hybrid,
- stronger replication,
- or negative-result framing.

## Expected First Artifact

```text
research/04_expert_placement_acceptance/optimize_expert_placement.py
```

It should consume Phase 03 router traces and produce placement maps plus
coverage metrics before implementing full acceptance measurement.

## Current Result

`results_qwen15_moe_placement_optimization.md` records the first placement
optimization run. Optimized/hot-replicated placement improves coverage only
slightly and still looks weak for naive local top-k drafting. The strongest
signal is high top-1 locality, suggesting a top-1 local draft variant.

Main-method constraint: local drafting should preserve the model's default
expert top-k and reroute that many experts into the local set. Cheaper top-1 or
top-2 drafts are separate variants.
