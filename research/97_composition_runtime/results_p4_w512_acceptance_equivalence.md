# Phase 97 P4 — boot-static w512 acceptance equivalence

Status: **PASS for using the exact boot-static w512 realization as an
acceptance-only surrogate; BLOCKED for GPU measurement, P4a engineering,
latency transfer, action admission, and performance claims.**

Date: 2026-08-09.

## Decision

The boot-static target-matching w512 realization has the same registered
attention read view, target-owned shared-KV requirement, and canonical
true-slot requirement as `target-matching-w512-masked-k4`. This clears the
current readiness item `w512_mask_equivalence_unproven`.

The result is deliberately one-way: a future matched boot-static run may
supply the w512 action's **acceptance**, but its latency is diagnostic and its
window cost receives no score credit. No runtime-masked action has been
implemented or admitted.

## Exact mask contract

The implementation is page-granular. With 16-token KV blocks, 16 sink tokens,
window 512, and K4, each draft query keeps:

```text
first sink page
union
last ceil((512 + draft_query_offset) / 16) pages
```

The four query offsets are `0, 1, 2, 3`. Step 0 additionally removes the
rejected suffix of width 0 through 4 after page compaction, exactly as the live
proposer does. Consequently, the contract is not approximated as a generic
512-token slice. For example:

| true sequence | query offset | rejected suffix | visible source intervals |
| ---: | ---: | ---: | --- |
| 528 | 0 | 0 | `[0, 528)` |
| 529 | 0 | 0 | `[0, 16) U [32, 529)` |
| 529 | 0 | 4 | `[0, 16) U [32, 525)` |
| 544 | 1 | 0 | `[0, 544)` |
| 545 | 1 | 0 | `[0, 16) U [32, 545)` |
| 20,480 | 0 | 0 | `[0, 16) U [19,968, 20,480)` |
| 20,480 | 1 | 0 | `[0, 16) U [19,952, 20,480)` |

The validator compares the checked-in compaction equations with an
independently stated sink-or-trailing-page predicate for every legal sequence
length from 1 through 20,480, all four K4 queries, and every legal step-0
rejected suffix. All **163,830** cases match. Their ordered coverage digest is
`6134578e29da4b30884c4de1ecfb6b2f7bcc338393e1a29f4d2334fa1b77d62a`.

Prefill remains full-KV. Transition events remain unscored. The surrogate
must use the pinned Qwen3-8B piecewise-chain path; the alternate scratchpad
FULLCG path is explicitly disabled so the proof cannot silently transfer
across a different attention realization.

## KV, slots, and weights

The proof binds the frozen minimal-B0 boot, action registry, runtime snapshot,
and relevant implementation symbols:

- all 36 draft attention layers require the exact target-twin KV tensor
  objects;
- one target-owned pool remains the only allocated KV pool;
- windowing replaces only the draft read-side block table and visible sequence
  length, leaving the source metadata, positions, and write-slot mapping
  unchanged;
- the target runner remains the source of canonical physical slots; and
- the draft weight path remains an exact target-parameter alias, with no
  alternate quantization.

CPU tests execute the live `_apply_draft_kv_window` method at short, threshold,
and maximum contexts, verify the exact physical page ids, and confirm that the
source block table, source sequence lengths, and slot tensor are unchanged.
They also exercise 36 exact KV aliases, stable true-slot backing storage, and
target-parameter aliases.

Live identity is still fail-closed: every future capture must repeat the
existing tensor-object, pool, slot-buffer, and weight-version checks. This
artifact proves equivalence of the two semantic contracts; it is not a GPU
tensor observation for an unimplemented runtime action.

## Claim boundary and readiness

| item | result |
| --- | --- |
| boot-static w512 acceptance surrogate | eligible under the exact contract |
| boot-static latency transfer | forbidden |
| window cost credit | forbidden |
| cost equivalence | not proven |
| live runtime-masked action | not implemented |
| action admission | false |
| GPU measurement authorization | false |
| P4a authorization | false |

The additive readiness chain is now:

```text
same_event_recorder_unwired             cleared previously
w512_mask_equivalence_unproven          cleared here
conservative_resource_bound_missing     remaining
```

The frozen preregistration and runner/scorer snapshots are not rewritten.
Only a separate approval after the conservative resource artifact exists may
authorize the matched GPU screen.

## Artifacts and validation

```text
schemas/p4_w512_equivalence.schema.json
data/p4/p4_w512_acceptance_equivalence.json
data/p4/p4_w512_acceptance_equivalence_validation.json
scripts/validate_p4_w512_equivalence.py
tests/test_p4_w512_equivalence.py
```

Validate with:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research/97_composition_runtime/scripts/validate_p4_w512_equivalence.py \
  --proof \
  research/97_composition_runtime/data/p4/p4_w512_acceptance_equivalence.json

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  research/97_composition_runtime/tests/test_p4_w512_equivalence.py -q
```

The focused suite passes 15 tests plus 7 subtests. The complete Phase 97 suite
passes 193 tests plus 21 subtests. Ruff check and format-check pass the two new
Python files. No GPU command was run.

## Next step

Build the conservative pre-engineering resource bound on top of the measured
minimal B0 candidate. It must account for the exact boot-static w512 capture
configuration without relabeling the older optimistic projection as an exact
post-capture measurement. That artifact may clear run readiness, but it still
must not self-authorize a GPU command.
