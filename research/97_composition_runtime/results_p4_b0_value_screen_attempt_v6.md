# Phase 97 P4 — B0 value-screen attempt V6

Status: **STOPPED during the first capture cell with zero complete captures.
V7 is consumed, the attempt is invalid for scoring, and its output cannot be
resumed or retried. The failure is a prefill-sample versus decode-only work
off-by-one in the measurement harness.**

Date: 2026-08-09.

## Result

The exact V7 invocation passed its source, virtualenv, Ninja, native-sampler,
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
```

The first R4 OFF cell progressed through all four eight-request microbatches.
The recorder then raised:

```text
P4 request 'R4-s0-p000' exceeded fixed output work
```

The parent stopped immediately.

```text
failed boot                              p4-b0-b1-p1-off
failed cell                     capture-b1-p1-off-r4-s0-r1
completed boots                                         0
complete captures                                       0
incomplete captures                                     1
adapted rounds                                          0
score emitted                                       false
```

The immutable machine record is
`data/p4/run_b0_value_screen_v6/failure.json`.

## Root cause

vLLM samples a request's first output token in its prefill event. P4 correctly
excludes that event from the decode-only `S_dec` measurement. With
`SamplingParams.max_tokens=512`, each request therefore supplied one
unmeasured prefill-sampled token and only 511 measured pure-decode commits.

The frozen recorder required 512 pure-decode commits for each of 32 requests.
The first cell consequently reached only `32 * 511 = 16,352` of its required
16,384 measured commits and never closed. The four prefill boundaries are the
missing capture step indices `512`, `1024`, `1536`, and `2048`.

The runner then began the next planned cell, which reused the same frozen
prompt ids. Step 2049 added a stale-cell commit to `R4-s0-p000` through
`R4-s0-p007`; the following step would take those requests beyond 512, so the
recorder failed closed before appending it. The preserved incomplete capture
therefore contains 512 commits for `p000` through `p007` and 511 for `p008`
through `p031`.

This is not a shared-KV, request-ID, model, driver, sampler, CUDA-graph, or GPU
capacity failure. It invalidates the measurement runner's fixed-work closure
and V7 run readiness. No K4 or W512 value result was reached.

## Fail-closed disposition

V7 authorized one attempt. That attempt is consumed. No fallback GPU, retry,
partial resume, adaptation, or scoring occurred. GPU 4 returned to zero active
compute processes. The incomplete capture, preparation files, and failure
record remain immutable at `data/p4/run_b0_value_screen_v6`.

## Required repair

Keep the decode-only recorder invariant and decouple generated work from
measured work:

1. request 513 total output tokens for every cell: one prefill-sampled token
   plus the frozen 512 pure-decode tokens;
2. keep the capture/scorer currency fixed at exactly 512 measured decode
   commits per request;
3. make the runner validate 513 frontend output tokens while the recorder
   closes at 512 pure-decode commits;
4. add fail-closed tests for the prefill-sample offset, exact closure, and
   transition to the next capture cell for OFF, K4, and W512; and
5. bind the repaired sources and a new create-only output path in a fresh V8
   authorization.

The V6 output must not be deleted, overwritten, or reused.
