# Phase 97 P4 — B0 value-screen attempt V3

Status: **STOPPED during the first OFF cell with zero complete captures. V4 is
consumed, this attempt is invalid for scoring, and its output cannot be resumed
or retried. No GPU command, P4a work, action admission, or performance claim is
currently authorized.**

Date: 2026-08-09.

## Result

The exact V4 invocation passed the virtualenv/Ninja and PyTorch-native-sampler
preflights and reached the first GPU-4 engine boot. Target and draft compilation
completed, self-spec weight sharing reported 291 aliased parameters, all 36
draft attention layers bound to target-owned KV, CUDA graph capture completed,
and the engine exposed 392,432 KV-cache tokens.

The first `R4` OFF cell then reached a mixed prefill/decode scheduler event.
The P4 recorder correctly rejected it because the frozen scorer accepts only
pure-decode, same-event measurements.

```text
failed boot                              p4-b0-b1-p1-off
failed cell                     capture-b1-p1-off-r4-s0-r1
completed boots                                         0
complete captures                                       0
empty create-new placeholders                           1
adapted rounds                                          0
score emitted                                       false
native sampler                                       pass
shared target KV                                     pass
root exception                 ineligible mixed scheduler event
```

The empty placeholder is not a capture. It was created fail-closed before the
first event and remains at zero bytes.

## Diagnosis

V4 fixed the CUDA/FlashInfer failure. The new failure is capture-ingress
geometry:

- the runner freezes `max_num_batched_tokens=8192`;
- speculative slot reservation reduces the effective scheduler budget to
  8,160 tokens;
- the first eight-prompt `R4` microbatch contains 65,200 prompt tokens, with
  individual lengths from 7,836 to 8,597; and
- the largest frozen microbatch across all regimes contains 112,908 prompt
  tokens.

The scheduler must therefore split prefill. Once an earlier request finishes
prefill, its decode step shares an event with another request's remaining
prefill. `build_same_event_record` labels that event
`prefill_or_mixed_batch`, and `P4SameEventRecorder` refuses it. This is the
right fail-closed outcome: ignoring the event would lose decoded tokens, while
recording it would charge prefill time to the decode-only objective.

The static conformance package did not check that each frozen prompt
microbatch fits the effective scheduler budget. This attempt therefore
invalidates run readiness, not the shared-KV, sampler, resource-floor, or value
hypotheses.

## Fail-closed disposition

The parent exited on the first non-zero child result. No retry, partial resume,
fallback GPU, adaptation, or scoring occurred. No engine or GPU compute process
remained after failure. The create-new V3 output is preserved as a consumed,
failed attempt and must never be deleted, overwritten, or reused.

The machine-readable record is
`data/p4/run_b0_value_screen_v3/failure.json`.

## Required repair

Keep the pure-decode recorder invariant. The minimal measurement-runner repair
is to make every submitted prompt microbatch complete prefill in one scheduler
event:

1. derive the maximum microbatch prompt total from the frozen manifest;
2. require an effective scheduled-token budget of at least 112,908;
3. account for the 32 speculative slots, so
   `max_num_batched_tokens >= 112940` (a rounded 114,688 is appropriate);
4. add fail-closed tests for the manifest-derived budget and pure-decode first
   captured event;
5. revalidate the 21,682-block shared-KV floor under the larger compile/input
   envelope; and
6. bind all changed sources and a new create-only output path in V5.

V4 is consumed and cannot authorize another GPU command.
