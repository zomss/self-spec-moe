# Phase 13: Affinity Placement + Request Scheduling + Gated Local Draft

Source phase: [`../11_intranode_ep_draft`](../11_intranode_ep_draft) (idea prompted
by Semantic Parallelism, arXiv:2503.04398)

## Objective

Test whether a Semantic-Parallelism-style co-scheduling can manufacture a
high-local-coverage *draftable subset* on **no-shared-expert** MoE models, so the
gated local draft ("draft only requests whose tokens use local experts") reaches
useful acceptance.

## Method

1. Decode real sequences; extract per-layer per-position true top-k experts.
2. **Affinity placement:** per layer, build the expert co-activation matrix and
   greedily cluster experts into `num_groups` balanced groups (collocate
   co-activated experts). Contiguous placement is the no-affinity baseline.
3. **Request-to-device scheduling:** assign each request to the group that
   maximizes its local coverage (oracle/best-case scheduling).
4. **Coverage + gating:** per-step local coverage distribution -> draftable subset
   (coverage >= threshold). Local Activation Rate (LAR) = mean coverage.
5. **Acceptance:** masked forward routing each request to its group's per-layer
   cluster; report acceptance for all steps and for the draftable subset.

Models: `Qwen/Qwen3-30B-A3B`, `openai/gpt-oss-20b` (no shared expert).
`num_groups` = the draft locality / "node" count. One-step acceptance proxy.

## Artifacts

| File | Purpose |
| --- | --- |
| `affinity_gated_draft.py` | Clustering + assignment + coverage + gated acceptance |
| `data/<model>_affinity.json` | Metrics by placement x num_groups |
| `results_affinity_gated_draft.md` | Findings |
