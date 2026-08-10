# Phase 97 P4 — B0 value-screen attempt V5

Status: **STOPPED at the first scheduler event with zero complete captures.
V6 is consumed, the attempt is invalid for scoring, and its output cannot be
resumed or retried. The failure is an exact request-identity normalization
mismatch, not mixed request ingress.**

Date: 2026-08-09.

## Result

The exact V6 invocation passed its source, virtualenv, Ninja, native-sampler,
in-process EngineCore, GPU identity/idle, and create-new-output checks. GPU 4
initialized the target-matching self-spec engine at the measurement-only
`114688 / 0.96` configuration:

```text
EngineCore                                      InprocClient
native PyTorch sampler                                 pass
target/draft weight aliases                            291
draft attention layers sharing target KV                36
private draft KV allocation                          false
actual shared-KV capacity                            22,090 blocks
required launch floor                                21,682 blocks
actual headroom                                         408 blocks
CUDA graph capture                                     pass
all eight requests queued before first step            pass
```

The first R4 OFF cell then raised:

```text
P4 event contains requests outside the frozen prompt cell:
['R4-s0-p000-aa61a533', ..., 'R4-s0-p007-97cdba3d']
```

The parent stopped immediately.

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
`data/p4/run_b0_value_screen_v5/failure.json`.

## Root cause

The frozen plan names requests by prompt record id, such as
`R4-s0-p000`. `LLMEngine.add_request()` passes each request through
`InputProcessor.assign_request_id()`, which retains that value as the external
id but changes the EngineCore id to `R4-s0-p000-<8 hex>` for internal
uniqueness.

The atomic-ingress proof knew about this representation and accepted either
the exact external id or an id beginning with `<external id>-`. The scored
recorder instead compared scheduler-row ids directly with the unsuffixed
`prompt_record_ids`. All eight correctly queued rows therefore failed its
exact-membership check before the first event could be serialized.

This failure does not invalidate the in-process queue-all mechanism, native
sampling, shared target KV, weight aliases, graph initialization, or resource
floor. It invalidates V6 run readiness because its proof and scored recorder
used different request-identity contracts.

## Fail-closed disposition

V6 authorized one attempt. That attempt is consumed. No fallback GPU, retry,
partial resume, adaptation, or scoring occurred. GPU 4 returned to zero
allocated MiB. The zero-byte placeholder, preparation files, and failure
record remain immutable at `data/p4/run_b0_value_screen_v5`.

## Required repair

Before another screen review:

1. define one canonical mapping from EngineCore ids to frozen external prompt
   ids, accepting only the exact known id or that id plus one exact eight-hex
   randomization suffix;
2. prove the mapping is one-to-one, rejects unknown suffixes and collisions,
   and emits frozen external ids to the capture/scorer;
3. use the same helper in the atomic-ingress proof and live recorder;
4. add an end-to-end live test that passes randomized EngineCore ids through
   the scored recorder; and
5. preserve the V6 output and seek a fresh source-bound V7 authorization.

Disabling request-id randomization globally is a possible diagnostic, but it
removes vLLM's internal uniqueness protection. Canonicalization at the
measurement boundary is the safer default repair.
