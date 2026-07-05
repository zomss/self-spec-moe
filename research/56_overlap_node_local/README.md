# Phase 56 — overlap x node-local: the ahead chain on the real 2-node fabric

**Source phases:** 32 (analytic overlap model: at high f the verify's comm
window hides the draft for free), 49 (OV0/OV1 machinery: side-stream ahead
chain, validation mode `VLLM_SELF_SPEC_AHEAD_CHAIN=1`, consume mode
`VLLM_SELF_SPEC_CONSUME_AHEAD=1`; validated single-node with a COMM-FREE
draft), 54/55 (node-local draft `VLLM_SELF_SPEC_DRAFT_NODE_LOCAL=1`: draft
MoE dispatch/combine over the intra-node `_EP_NODE`/`_DP_NODE` subgroup;
236B 2-node K=1 lockstep 81.0/273.4/378.6 tok/s at b8/32/64, accept ~1.9).

**Objective.** Compose the OV1 ahead chain with the node-local draft on the
real h107+h106 fabric, staged:

- **Stage A (physics):** does a draft that itself issues REAL intra-node
  collectives (side stream, NVLink) overlap with the verify's inter-node EP
  collectives (main stream, RoCE), or do they serialize? Measured as the
  cycle-time delta of AHEAD_CHAIN=1 (validation mode: ahead AND lockstep
  propose both run) vs off, plus serialized component attribution via the
  self-spec profiler (whose region syncs SERIALIZE the overlap — profiled
  runs are attribution-only, never the A/B).
- **Stage B (only if A shows meaningful hiding):** node-safe consumption —
  the consume/propose decision made uniform across each node's ranks.

**Code facts established up front (see results):**
- The ahead chain honors the node-local draft config: its forwards pass
  `additional_kwargs=self._draft_forward_additional_kwargs`
  (= `{SELF_SPEC_NODE_LOCAL_KEY: True}` under the flag), so ahead forwards
  issue the same node-subgroup collectives as propose.
- OV1(a) stash gate requires `num_speculative_tokens > 1` (the ahead chain
  CONTINUES the decode-shaped chain, which only exists at K>=2) — so Stage A
  runs at **K=2** (unlocked at 236B by Phase 55 3c), not the K=1 ladder
  config.
- AHEAD_CHAIN requires the phase-49 isolation stack
  (`VLLM_SELF_SPEC_DRAFT_GRAPH_POOL=1` + `VLLM_SELF_SPEC_DRAFT_WORKSPACE=1`),
  not present in env_2node.sh — set in BOTH arms of every A/B here.
- Validation mode is symmetric only in uniform-decode steady state: the
  batch-boundary fence is RANK-LOCAL, so a collectivizing draft can diverge
  at stagger/drain steps (the Phase 54/55 deadlock class). The harness's
  identical fixed-length batches on every rank keep boundaries synchronized;
  wedges are a measured outcome, not an assumption.

## Commands

```
bash research/56_overlap_node_local/scripts/run_smoke_v2lite.sh   # Stage A smoke (control + ahead)
bash research/56_overlap_node_local/scripts/run_236b_ahead.sh     # Stage A physics at 236B
```

## Decision criteria

Stage A GO: with ahead work ~1 draft forward (W7_AHEAD_STEPS=1), cycle time
grows <20% (mostly hidden); NO-GO: grows by ~a full draft forward
(serialized). Either result is the publishable measurement. Stage B gate:
meaningful hiding AND the draft-forward floor small enough that the consume
arithmetic can win.

## Artifacts

`results_overlap_node_local.md`, `logs/`, `data/`.
