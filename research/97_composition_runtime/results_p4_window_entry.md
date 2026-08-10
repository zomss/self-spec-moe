# Phase 97 P4 entry — w512 masked window

Status: **PASS as a fail-closed preregistration; NOT ADMITTED for live
selection, GPU measurement, or a performance claim. One action is selected:
`target-matching-w512-masked-k4`. It is acceptance-only, receives no window
cost credit, reuses the P3 K4 graph descriptors, and remains blocked on
runtime transport, exact live identity, matched controls, transition cost,
an exact post-window-capture B0 resource record, and a Phase 96/F value gate
that is now explicitly `not_approved`.**

Date: 2026-08-09.

## Decision

P4 will start with one `w512` masked view over the target-matching K4 action:

| field | registered value |
| --- | --- |
| action id | `target-matching-w512-masked-k4` |
| proper subset | `target-matching-k4` |
| window / sinks | 512 / 16 tokens |
| grade | `masked`, `acceptance_only` |
| cost credit | forbidden |
| new graph | none |
| K / skip / weight path | 4 / none / target-matching |
| transition cost | unmeasured |
| status | `registered_not_admitted` |

This is the minimum P4 mechanism. The action changes the draft's visible
history but not its cache owner, allocation, true slots, weight path, skip
set, K, target graph, or draft graph. A cost-true per-window graph is not
registered.

## Why w512, and what the evidence does not prove

The package binds every source by SHA-256 and separates selection evidence
from promotion evidence:

- W6 repeatedly values `w512`, including the best static dense composition,
  but those measurements used a quantized draft. They select a useful window
  value; they do not transfer acceptance to target-matching B0.
- W13 shows that window cost leverage is negligible at low total KV and
  material only at high total KV. It explicitly classifies the low-KV window
  as an acceptance lever and also shows that the sync-bracketed profiler
  cannot price this new realization.
- W14/B certifies separately booted cost transfer in all 6/6 strata. It does
  not price masked `w512` in the P3 graph; registered directional containment
  remains 6/12.
- `w14_interval_fix.json` is retained as the post-hoc 12/12 containment
  diagnostic with zero elimination flips. It is not treated as a registered
  rescore or as Phase 96/F resident-portfolio evidence. The subsequent F
  audit records `not_approved`, with value unresolved rather than disproven.

Therefore `phase96_exact_action_match=false`. Acceptance must be measured for
this exact target-matching masked action, and separately booted `w512` cost
must not be assigned to it.

## Shared-KV and boundary contract

The validator derives a proposed three-action registry from the hash-pinned
P3 package and overlays the window action on the P3 runtime snapshot. The
derived package passes the existing shared-KV registry and runtime validators:

- all **36/36** draft attention layers retain their target-twin binding and
  exact-storage requirement;
- allocation remains one target-owned pool;
- K4 and masked w512 use the canonical target true-slot mapping; and
- OFF changes no cache ownership and leaves the target pool allocated.

This is schema/synthetic entry evidence, not a live tensor proof for the new
action. Live wiring must repeat exact storage identity and canonical slot
identity before promotion.

P3b's boundary policy is immutable in the entry package: one action covers a
whole decode dispatch; switching occurs only at a target-step boundary; mixed
prefill/decode admission aborts pending drafts, executes existing decodes as
q=1/OFF, counts discarded rows as neither armed nor accepted, and permits
re-entry only after a later pure-decode OFF step.

## Frozen controls

The first confirmation is exactly a three-way comparison:

1. `off`;
2. `target-matching-k4`; and
3. `target-matching-w512-masked-k4`.

K4 is the window action's proper subset. Comparisons use target-local
correctness and match target/draft versions, boot, shared-KV binding, hardware,
parallel layout, workload trace, batch, context, generated suffix, K, kernel,
graph grade, warmup, and measurement currency. Acceptance is action-specific.
Cross-boot AR identity remains a non-gating BF16 diagnostic.

## Resource decision

The old exact minimal-B0 measurement reports 24,529 shared-KV blocks against
the engineering envelope's 21,000-block requirement. It is copied only as an
optimistic proxy:

```text
candidate_realization_match = false
measurement_relation = optimistic_proxy_ceiling
planner decision = reject
reason = capacity_evidence_not_exact
```

Even though the masked design adds no resident graph, its runtime metadata,
buffers, capture state, and integrated allocation have not been measured. The
3,529-block apparent headroom is not an admission result. After implementation,
the exact action pool must be booted and captured, then peak HBM and actual
shared-KV blocks must be recorded in a new `measured_exact` candidate.

## Promotion gates

| gate | state |
| --- | --- |
| P3b mixed-boundary abort | pass |
| Phase 96/F resident value | not approved |
| scheduler-to-proposer action transport | pending |
| live 36-alias and true-slot proof | pending |
| target-local OFF/K4/w512 controls | pending |
| exact post-capture B0 resource candidate | pending |
| transition/probe overhead | pending |
| cost-true claim | forbidden for this action |
| private or quantized draft KV | forbidden |

No pending or failed gate is inferred as passed by the entry validation. The
machine-readable F evidence audit is summarized in
`results_p4_f_value_gate.md`.

## Artifacts and command

```text
schemas/p4_window_entry.schema.json
data/p4/p4_window_entry_w512_masked.json
data/p4/candidate_b0_window512_masked_projection.json
data/p4/p4_window_entry_validation.json
scripts/validate_p4_window_entry.py
tests/test_p4_window_entry.py
```

Validate with the repository environment:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research/97_composition_runtime/scripts/validate_p4_window_entry.py \
  --entry \
  research/97_composition_runtime/data/p4/p4_window_entry_w512_masked.json

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research.97_composition_runtime.tests.test_p4_window_entry -v
```

The focused suite contains 17 positive and fail-closed tests; the full Phase
97 CPU suite passes 86 tests. Ruff 0.14.0 passes the new Python files. No GPU
command is authorized by this entry result.

## Next artifact

P4a is not the next artifact because Phase 96/F did not approve its value. The
next artifact is a matched target-matching B0 value-screen preregistration
after the workload objective and weights are frozen and target-step accounting
is repaired. If that screen later authorizes engineering, P4a may transport
this exact action id and implement same-graph masking. A separately registered
confirmation would still need to prove live aliases, true slots, target-local
correctness, matched controls, transition overhead, and exact post-capture
capacity. Until then, the executable registry remains P3's two actions only.

Follow-up: the preregistration is complete in
`results_p4_b0_value_screen_preregistration.md`, but it is intentionally
run-blocked. The executable registry and every P4a authorization remain
unchanged.
