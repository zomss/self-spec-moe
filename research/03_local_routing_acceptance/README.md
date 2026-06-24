# Phase 03: Local-Routing Acceptance

Source phase: [`../02_timing_envelope_experiment`](../02_timing_envelope_experiment)

## Objective

Measure whether local-only MoE routing can produce draft tokens with acceptance
high enough to realize the communication envelopes from Phase 01 and Phase 02.

The phase asks:

```text
Can local expert routing reach beta high enough for useful Self-MoE-spec speedup?
```

## Assumptions

- Timing alone is insufficient; the remaining critical unknown is draft quality.
- Verification remains exact full-routing MoE, so final generation can remain
  lossless if standard speculative verification is used.
- The first pass may use offline simulation and routing/logit analysis before a
  runtime local-draft implementation exists.

## Target Metrics

| Metric | Meaning |
| --- | --- |
| `gamma` | Probability mass or gate weight covered by local experts. |
| top-k overlap | Fraction of true routed experts available locally. |
| `beta` | Draft-token speculative acceptance rate under local-only routing. |
| acceptance by position | Acceptance at draft positions `1, 2, 4, 8`. |
| placement sensitivity | Difference across random, locality-aware, hot-expert, and shared-expert placement. |

Terminology:

```text
expert_top_k = number of experts selected by the MoE router
num_spec_tokens = number of speculative draft tokens before verification
```

Use explicit names to avoid overloading `k`.

## Initial Experimental Setting

| Dimension | Target |
| --- | --- |
| Model | Small/available MoE first, then target DeepSeek/Qwen-style MoE if available |
| Hardware | Single node is sufficient for offline simulation |
| Draft lengths | `k = 1, 2, 4, 8` |
| Local expert placements | random, contiguous/EP-style, hot-expert replicated, locality-aware if traces exist |
| Prompt set | short deterministic prompts first; later domain buckets such as chat/code/math |
| Sampling | greedy first for reproducibility; sampling later if needed |

## Method

1. Run the target MoE normally and collect router logits/top-k expert choices.
2. Simulate local expert placement for each EP rank.
3. Mask routing to local experts and compute:
   - local gate mass `gamma`,
   - true top-k overlap,
   - per-layer route divergence.
4. Generate local-only draft tokens, then verify with full routing.
5. Measure acceptance rate by draft position and placement strategy.

## Target Outputs

All outputs should stay in `research/03_local_routing_acceptance/`.

| Output | Path | Purpose |
| --- | --- | --- |
| Simulation script | `simulate_local_routing.py` | Offline local-routing and acceptance simulation |
| Router extractor | `extract_router_logits.py` | Extract real router logits from a Hugging Face MoE checkpoint |
| Acceptance proxy | `compare_masked_router_distribution.py` | Compare full-router and local-masked next-token distributions |
| Input prompts | `data/prompts_*.jsonl` | Prompt sets used for routing/acceptance runs |
| Routing traces | `data/routing_traces_*.jsonl` | Router logits/top-k summaries if saved |
| Metrics CSV | `data/local_routing_metrics_*.csv` | `gamma`, top-k overlap, and acceptance metrics |
| Summary | `results_*.md` | Interpretation and go/no-go for runtime work |
| Figures | `figures/*.png` or `figures/*.pdf` | Acceptance vs `gamma`, placement, and draft position |

## Go/No-Go Criteria

Proceed toward a runtime local-draft implementation only if:

- `beta >= 0.8` in the high exposed-communication regime, or
- `beta >= 0.9` for moderate exposed communication / DBO-composed settings, and
- acceptance does not collapse sharply at draft positions `2-4`.

Pivot if:

- local-only routing acceptance is far below `0.8`,
- `gamma` is low for realistic placements,
- acceptance requires unrealistic hot-expert replication,
- draft quality decays too quickly with `k`.

## Expected Next Artifact

The next artifact is:

```text
research/03_local_routing_acceptance/simulate_local_routing.py
```

Start with a minimal script that records router top-k overlap and `gamma`; add
full draft-token acceptance after the routing-only measurement is stable.

## Current Result

`results_initial_synthetic_routing.md` records the first synthetic routing
coverage validation. It confirms that uniform routing is weak, while hot/local
expert concentration can make `gamma` and top-k overlap much higher. Real router
logits are needed next.

`results_powermoe3b_real_routing.md` records the first real-router-logit run on
`ibm-research/PowerMoE-3b`. Its simple-placement coverage is weak, suggesting
that this model is not a strong Self-MoE-spec candidate without more advanced
placement or replication.

`results_qwen15_moe_real_routing.md` records the Qwen1.5-MoE real-router run.
It also shows weak local coverage under simple placements, especially as EP size
increases.

`current_conclusion.md` summarizes the cross-model Phase 03 conclusion and
recommends pivoting toward cheaper/replicated/hybrid drafts unless direct
acceptance proxy results are unexpectedly strong.

`results_qwen15_moe_acceptance_proxy.md` runs the direct full-router vs
local-masked next-token distribution proxy. It shows poor overlap for
Qwen1.5-MoE EP2 and argues against naive local-top-k runtime implementation.

Phase 04 has been initialized at
[`../04_expert_placement_acceptance`](../04_expert_placement_acceptance) to tune
expert placement and measure actual speculative acceptance.

See [`../status.md`](../status.md) for the consolidated project conclusion.
