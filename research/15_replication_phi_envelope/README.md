# Phase 15: G>=4 Replication (EPLB / Expert-Cache) Speedup Envelope

Source: [`../13_affinity_placement_gated_draft`](../13_affinity_placement_gated_draft),
[`../11_intranode_ep_draft`](../11_intranode_ep_draft), [`../09_rebalancing_ceiling`](../09_rebalancing_ceiling)

## Objective

G=2 acceptance works (Phase 13) but G>=4 fails on placement alone. Test whether
**EPLB / expert-cache replication** can reach G>=4 acceptance without erasing the
communication advantage, and find the optimal replication level + break-even
inter-node latency.

## Method

Acceptance(phi) from the Phase 09 mass-optimal cache sweep (M = phi*E). Latency-
bound comm model (measured NVLink 30us/collective from Phase 12; modeled IB;
2*num_layers collectives/step). Key point: at decode the all-to-all is latency-
bound, so replicating to fraction phi raises acceptance while the verify inter-node
collective still fires -> the draft still removes its full latency. Speedup is then
monotonic in phi; push to the memory budget.

```bash
.venv/bin/python research/15_replication_phi_envelope/phi_envelope.py
```

## Result

See `results_phi_envelope.md`. Replication makes G=4 viable, but only when exposed
inter-node collective latency is >= ~100-200us. Qwen3 nets ~1.18x at phi=0.5
(R~=2x, ~8 experts/GPU, acceptance 0.78) iff IB latency >= ~200us/collective.
The sole remaining gate is the real exposed inter-node A2A latency after
DeepEP/DBO overlap, measurable only on multi-node IB.
