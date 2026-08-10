# Phase 97 P3b — mixed-boundary K4 abort repair

Status: **PASS. Pending K4 rows are now aborted before a mixed
prefill/decode target pass, existing decodes run at q=1/OFF, discarded rows
contribute neither armed nor accepted counts, and K4 is re-armed only at a
later pure-decode boundary. The P3-specific correctness blocker for P4 is
cleared. No performance claim is made.**

Date: 2026-08-09.

## Frozen contract

Phase 97 uses target-local speculative correctness:

- an accepted draft must equal the argmax of the target verifier row that
  accepts it;
- a replacement or bonus token must come from that same target forward; and
- cross-boot sequential-AR identity is diagnostic unless the target
  realization is separately proven query-width invariant.

At a mixed admission boundary, the scheduler now applies this fixed policy:

1. discard every pending row from the complete K4 dispatch;
2. replace its per-request provenance with OFF;
3. reduce each existing decode query from q=5 to q=1 before constructing the
   runner input;
4. dispatch OFF for the following draft action; and
5. permit K4 re-entry only on a later pure-decode target-step boundary.

The trace fields have exact meanings:

- `aborted_action_id` names the already-produced action whose rows were
  discarded;
- `discarded_draft_width` is the discarded width per affected request, not an
  aggregate; and
- `aborted_draft_request_count` is the number of discarded rows.

A non-abort record must carry `null/0/0`. The aggregate discarded token count
is width times request count.

## Implementation and fail-closed checks

The repair is applied after admission is known and before `SchedulerOutput`
is built. It mutates only scheduled query widths, verifier draft rows, and
action provenance; target-owned KV allocation and true slot mapping do not
change.

The scheduler and worker reject:

- a partial-row abort;
- a width or provenance other than one complete K4 dispatch;
- mixed K4 verification without an abort;
- any prefill/mixed pass that does not select OFF with `force_off` intent;
- abort metadata that does not resolve to q=1/OFF; and
- scheduler/worker disagreement about any abort field.

Passive accounting records an aborted mixed step as replay-ineligible. Its
discarded proposals are absent from both `D_armed` and `A_accepted`.

## CPU validation

The focused contract and real scheduler-admission test passed:

```text
32 passed
```

The cumulative Phase 97 suite passed:

```text
82 passed, 14 subtests passed
```

The exercised scheduler case starts with one decoded request carrying four
pending drafts, admits a ten-token prefill, and proves that the emitted
scheduler geometry changes from the provisional q=[5,10] plan to q=[1,10]
with no verifier draft rows.

Ruff 0.14.0 passed on the changed runtime, scheduler, tests, and Phase 97
harness files. Python compilation also passed.

## Live GPU validation

The non-profiled correctness matrix ran on H100 GPUs 4–5 with Qwen3-8B,
BF16 target/KV, V1, native sampling, compiled target execution, aliased
target-matching draft weights, and one target-owned shared KV cache.

| run | purpose | result |
| --- | --- | --- |
| dynamic K4/OFF | repaired K4 -> abort/OFF -> K4 path | pass |
| OFF with resident K4/OFF graph pool | q=1 boundary control | pass |
| fixed K4 | stable-decode control | pass |
| sequential AR | non-gating numerical diagnostic | reported mismatch |

The dynamic trace has 13 records, 11 replay-eligible records, and one abort at
engine step 4. That record proves:

```text
verified action             off
next action                 off
selection intent            force_off
aborted action              target-matching-k4
discarded width / rows      4 / 1
target query widths         [1, 10]
draft widths                [0, 0]
query starts                [0, 1, 11]
long/short positions        [35] / [0..9]
D_armed / A_accepted        0 / 0
E_committed                 1
```

The long target slot is disjoint from all ten short-request slots. Engine
step 7 is the later pure-decode OFF-to-K4 bootstrap: its target query width
and draft step-0 width are both one, and it produces four drafts. Subsequent
steps verify K4.

All dynamic K4 records passed the target-local check: their accepted count is
exactly the prefix length for which draft ids equal the target verifier's
argmax ids. The fixed-K4 control matches the steady K4 inputs, positions,
draft ids, and verifier argmax ids.

The shape-matched OFF control retains an unreachable K4 schedule entry so it
reserves the same K4/OFF graph pool while selecting only OFF at runtime. At
the boundary it matches the repaired run's execution mode, input ids,
positions, sequence lengths, q=[1,10] geometry, logit-row selection, and
top-eight candidate sets.

The two controls still choose different long-request argmax ids at output
index 12:

| realization | token id | top-two logits |
| --- | ---: | --- |
| repaired K4-history boundary | 22406 | 25.875, 25.750 |
| fixed-OFF-history control | 5005 | 25.875, 25.875 |

The maximum matched top-logit delta is 0.125, one BF16 step. Both outputs
equal the argmax of their own target forward. The histories contain the same
tokens but their committed target KV was produced under earlier q=5 versus
q=1 target shapes, so this is the already-registered execution-shape
numerical diagnostic, not an abort, slot, or shared-KV failure. Sequential AR
also first differs at index 12 and remains non-gating for the same reason.

Exact live invariants remain intact:

- all 36 draft attention layers alias target KV tensors;
- all 291 draft parameters alias target parameters;
- binding, target pool, and canonical true-slot identities are stable;
- no private draft-KV pool is active; and
- eligible dynamic totals close with `H=13`, `D=7`, `A=28`, `C=1`, and
  `E=40`, so `E+C=A+H`.

GPUs 4–5 returned to zero allocated MiB after shutdown.

## Decision

P3b passes and the mixed-boundary correctness blocker is removed. P4 may now
prepare one preselected runtime-window action, subject to the existing Phase
96 evidence, exact post-capture resource gate, mechanism-retention check, and
matched non-regression control. This result does not authorize a Cartesian
window/skip/weight pool, B1 co-residency, or performance claims.

## Artifacts

- `data/p3/p3b_mixed_abort_20260809_dynamic.json` and its `_trace.jsonl`;
- `data/p3/p3b_mixed_abort_20260809_off_pool_ctx35.json` and its
  `_trace.jsonl`;
- `data/p3/p3b_mixed_abort_20260809_k4.json` and its `_trace.jsonl`;
- `data/p3/p3b_mixed_abort_20260809_ar.json`;
- `data/p3/p3b_mixed_abort_20260809_verification.json`; and
- `scripts/verify_p3b_mixed_abort.py`.
