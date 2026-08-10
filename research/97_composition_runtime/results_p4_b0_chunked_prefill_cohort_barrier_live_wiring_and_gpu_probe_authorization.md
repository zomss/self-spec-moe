# Phase 97 P4 — cohort-barrier live wiring and GPU-probe authorization

Date: 2026-08-09

Status: **CPU PASS; the V1 non-scored probe authorization was CONSUMED by a
pre-child path-normalization failure; no GPU model executed**

## Outcome

The bounded chunked-prefill cohort barrier is now wired into the synchronous
engine behind an explicit measurement-only request marker. Ordinary serving
does not create or consult a barrier. The live CPU execution path passes for
OFF, K4, and the boot-static w512 surrogate.

A separate source-bound package authorized exactly one non-scored physical
boot on GPU 4. Its later exact invocation created the output directory but
failed before child launch because the relative reviewed path was not
normalized. Nothing was scored and no V10 authority was created.

## Live request contract

The capture runner writes `p4_capture_cohort_barrier` into
`SamplingParams.extra_args`. The closed payload carries:

- contract ID and `measurement_only=true`;
- one stable cohort ID;
- the exact ordered prompt-slice request IDs;
- the boot action ID;
- one unmeasured prefill sample per request; and
- the exact measured decode-token target.

Malformed, partial, reordered, duplicated, foreign, streaming, pooling, async,
or non-8192 chunked-prefill requests fail before measured execution. The marker
also requires the live K/OFF path and a capture recorder; it cannot silently
activate in ordinary serving.

## Synchronous scheduling behavior

Admission registers every exact member and seals only after the complete
frozen slice is queued. At each scheduler boundary the barrier receives the
state of every unfinished member.

When a request samples its one prefill token before its peers, the scheduler
retains its target-owned shared KV, skips it in subsequent prefill allocation,
and preserves its boot-action draft tokens. K4 is armed on that request's
final prefill event; later prefill events cannot overwrite the held drafts.

Once every member has one prefill sample, the first measured event must contain
every unfinished cohort member, in frozen order, as pure decode. OFF requires
query width 1. K4 and w512 require query width 5; w512 remains the validated
boot-static masked K4 realization. The same-event commit, invalid-token,
preemption, recomputation, and frontend-output counters close through the
existing barrier before the capture recorder accepts the event.

Member cancellation expands to every unfinished member. Identity drift,
missing members, foreign scheduling, preemption, recomputation, invalid
speculation, non-atomic release, no progress, early finish, or incomplete
shutdown aborts and cleans the complete active cohort.

## CPU execution-path proof

The focused live regression uses five prompts whose aggregate prefill exceeds
8192 tokens. Four requests complete first and are held; the fifth completes in
the next chunk. The following event atomically releases all five.

The proof passes independently for:

- OFF: five q=1 rows;
- target-matching K4: five q=5 rows with preserved K4 provenance; and
- w512: five q=5 rows with the same target-owned KV and masked K4 realization.

Additional regressions cover whole-cohort cancellation, incomplete shutdown,
ordinary-serving bypass, exact runner marker transport, and one-parent/
one-child probe dispatch.

## Non-scored GPU-4 probe authority

The authorization is
`data/p4/p4_b0_chunked_prefill_probe_authorization.json`. It is bound to the
live scheduler, barrier, GPU worker, capture and probe runners, tests, prompt
manifest and bundle, prior CPU proof, transient-bound diagnosis, and consumed
V9 failure.

The one allowed boot uses:

- physical GPU 4 only, UUID
  `GPU-c9d19019-5065-2353-80a9-f1797eb19d51`;
- compiled target-matching K4;
- `max_num_batched_tokens=8192`, `max_num_seqs=32`, and
  `gpu_memory_utilization=0.90`;
- one eight-prompt seed-0 cohort from each of R4, R5, and R5cot; and
- one unmeasured prefill token plus one non-scored measured decode token.

It passes only if the boot records positive **actual** CUDA graph memory,
retains at least 21,682 shared target-KV blocks, observes a pure first measured
decode for all three cohorts, and reports zero preemption, recomputation, and
invalid speculative tokens.

Retry, resume, V9 capture reuse, fallback GPU, scoring, P4a work, action
admission, performance claims, and V10 are all forbidden. Any source drift
invalidates the package before output creation.

## Later execution outcome

The exact V1 invocation was attempted once. Native-sampler, in-process
EngineCore, and GPU-4 identity/idle preflights passed. After creating the
reviewed output directory, the parent called `relative_to(REPO_ROOT)` on the
still-relative output `Path` and raised `ValueError` before writing
`preparation.json` or spawning the child.

The authorization and output path are consumed. No model boot, cohort,
target event, probe result, or score exists. The immutable failure is recorded
in `data/p4/run_b0_chunked_prefill_probe_v1/failure.json`.

## Validation

```bash
.venv/bin/python -m pytest \
  tests/v1/core/test_scheduler.py -q -k p4_cohort

.venv/bin/python -m pytest \
  research/97_composition_runtime/tests/test_p4_b0_capture_runner.py \
  research/97_composition_runtime/tests/test_p4_b0_chunked_prefill_probe_authorization.py \
  -q

.venv/bin/python \
  research/97_composition_runtime/scripts/validate_p4_b0_chunked_prefill_probe_authorization.py
```

Focused results: 38 K/OFF runtime tests, six live scheduler cohort tests, and
58 proof/runner/authorization tests plus five subtests pass. The complete
Phase 97 audit reports 464 passed and 43 subtests passed. Its only two failures
are the untouched historical V9 run-readiness assertions, which correctly
reject the consumed, source-stale V9 package. Ruff check and format pass the
changed Python files. These were pre-execution review results.

## Next gate

The path repair and exact registered-relative-argv regression passed, but the
later V2 execution stopped after engine boot when the live guard confused the
configured 8,192-token bound with the expected 8,160-token effective budget.
V2 is consumed without a cohort, result, or score. The real-config regression
and guard/evidence repair now pass, and a fresh source-bound V3 package
authorizes one unexecuted non-scored GPU-4 probe at an absent output. V10
remains unauthorized.
