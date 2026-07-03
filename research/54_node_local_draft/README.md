# Phase 54: node_local_draft — hierarchical draft (intra-node EP over NVLink)

**Source phases:** 53 (236B at EP16: device-local draft coverage 6.25% ->
accept ~1.3, below the f~0.7 win bar ~1.6-1.8; V2-Lite in-engine slope:
accept 1.88 @50% coverage, 1.36 @12.5%), 11/19 (the hierarchical design:
draft = intra-node EP only, verify = full EP; Phase 12: intra-node A2A ~free),
52 (fabric + engine groundwork).

**Objective.** Implement the Phase 11/19 node-local draft in the engine:
the draft routes to any expert resident on ITS NODE (2 nodes -> 50% coverage)
and dispatches over an intra-node EP subgroup (NVLink only; the AgRs backend's
comm volume is routing-independent, so node-local routing REQUIRES a
node-scoped collective to save anything). Verify unchanged (global EP16).

**Prediction (pre-registered).** Coverage 50% -> beta ~0.85-0.9 (measured
V2-Lite slope) -> accept(K=1) ~1.85-1.9 vs win bar ~1.6-1.8 at f~0.7:
**expected first real end-to-end 2-node WIN**. Draft comm cost: ~118 colls x
~30-60us NVLink ~ 4-7 ms/forward (vs ~30-40 ms saved inter-node comm).

**Design sketch.**
- Env: `VLLM_SELF_SPEC_DRAFT_NODE_LOCAL=1` (default off; implies/extends
  DRAFT_LOCAL_ROUTE semantics from device-local to node-local).
- Routing mask: allow experts whose owner EP rank is on this node (derived
  from the expert placement map + ranks-per-node), softmax renorm as in
  device-local.
- Dispatch: node-scoped AgRs (all_gather + reduce_scatter over an intra-node
  EP subgroup created at init; NVLink). Draft-forward only, keyed off the
  forward-context self-spec key (same mechanism as SELF_SPEC_LOCAL_ROUTE_KEY).
- DP coordination: draft needs none across nodes (Phase 52's
  DRAFT_SKIP_DP_COORD applies); within-node the subgroup collective itself
  is the sync.

**Steps.** (1) Read the local-route + AgRs plumbing; (2) implement env-gated;
(3) V2-Lite 1-node sanity (node-local == full EP when nodes=1: accept must
match EP-full ~= plain accept ceiling); (4) 2-node V2-Lite (accept vs the
EP2 1.88 point); (5) 2-node DeepSeek-V2 236B spec vs the Phase 53 baseline.

**Expected artifact.** `results_node_local.md` + engine changes on
`research/self-spec-moe`.
