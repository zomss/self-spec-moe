# Phase 14: Intra-node Semantic-Parallelism Pattern

Source phase: `13_affinity_placement_gated_draft`

## Objective

Check whether our MoE routing traces show the same intra-node qualitative pattern
as Speculative MoE / Semantic Parallelism: affinity-aware expert placement plus
request-to-group scheduling should raise local activation rate (LAR) and reduce
remote expert-routing volume.

This phase is intentionally narrower than an end-to-end speedup claim. It tests
the routing/communication-volume pattern that should precede any runtime
implementation.

## Assumptions

- Phase 13 JSON files are the source of truth.
- `contiguous_G*` is the baseline placement.
- `affinity_G*` is the Semantic-Parallelism-style placement plus oracle
  request-to-group assignment.
- Remote A2A volume is proxied by `1 - LAR`; this is a volume proxy, not a
  measured vLLM latency.
- Acceptance values are one-step sampled proxies from Phase 13.

## Commands

```bash
.venv/bin/python research/14_intranode_semantic_pattern/summarize_intranode_pattern.py
```

Outputs:

- `data/intranode_semantic_pattern.csv`
- `data/intranode_semantic_pattern.json`
- `figures/lar_comparison.svg`
- `results_intranode_semantic_pattern.md`

## Decision Criteria

The paper-like intra-node pattern is present if affinity placement:

1. Increases LAR over contiguous placement.
2. Reduces remote-volume proxy `(1 - LAR)` by a large fraction.
3. Does so most strongly at coarse intra-node grouping (`G=2`), where acceptance
   remains useful enough for a gated draft discussion.

## Expected Next Artifact

If this phase confirms the locality pattern, the next artifact should be a
runtime-facing microbenchmark: measure the actual all-to-all or all-to-allv
latency against controlled LAR on the target intra-node fabric, then decide
whether a vLLM prototype is justified.

