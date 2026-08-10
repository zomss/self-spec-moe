# Phase 97 P4 — B0 value-screen attempt V4

Status: **STOPPED at the first live OFF event with zero complete captures. V5
is consumed, the attempt is invalid for scoring, and its output cannot be
resumed or retried. The exact eligibility exclusion was not persisted.**

Date: 2026-08-09.

## Result

The exact V5 invocation passed its virtualenv, Ninja, native-sampler, source,
GPU, and create-new-output checks. GPU 4 initialized the target-matching
self-spec engine at the measurement-only `114688 / 0.96` configuration:

```text
native PyTorch sampler                              pass
target/draft weight aliases                         291
draft attention layers sharing target KV             36
private draft KV allocation                       false
actual shared-KV capacity                         22,090 blocks
required launch floor                             21,682 blocks
actual headroom                                      408 blocks
CUDA graph capture                                  pass
```

The first R4 OFF cell then raised
`P4 capture refuses an ineligible live event`. The strict recorder had already
created its fail-closed placeholder but rejected the event before serializing
it. The parent stopped immediately.

```text
failed boot                              p4-b0-b1-p1-off
failed cell                     capture-b1-p1-off-r4-s0-r1
completed boots                                         0
complete captures                                       0
empty create-new placeholders                           1
adapted rounds                                          0
score emitted                                       false
```

The immutable machine record is
`data/p4/run_b0_value_screen_v4/failure.json`.

## What this invalidates

The full-prefill budget removed the known 8,160-token ingress limit: the first
microbatch has 65,200 prompt tokens and fits inside the 114,656-token effective
scheduler budget. Nevertheless, `score_eligible` was false. The generic
exception does not reveal whether the event carried:

- `prefill_or_mixed_batch`;
- a preemption;
- recomputation; or
- invalid speculative tokens.

Therefore the earlier claim that budget repair alone preserved a pure-decode
first captured event is disproven. This attempt does not invalidate native
sampling, shared target KV, weight aliasing, graph initialization, or the
resource floor; it invalidates V5 run readiness and leaves capture ingress
undiagnosed.

## Fail-closed disposition

V5 authorized one attempt. That attempt is consumed. No fallback GPU, retry,
partial resume, adaptation, or scoring occurred. GPU 4 returned to zero
allocated MiB. The zero-byte placeholder and all preparation files remain
immutable at `data/p4/run_b0_value_screen_v4`.

The separately authorized `8192 / 0.90` real-serving diagnosis is reported in
`results_p4_b0_serving_chunked_prefill_diagnosis.md`. Its PASS establishes the
serving mixed-boundary behavior but does not repair or authorize the value
screen.

## Required repair

Before any V6 screen authorization, persist the exact exclusion reasons for a
non-scoring full-prefill reproduction (or include them in the fail-closed
exception), identify the first event geometry, and add a live ingress gate
that proves the first scored event is pure decode. Keep the current V4 output
and V5 authorization unchanged.
