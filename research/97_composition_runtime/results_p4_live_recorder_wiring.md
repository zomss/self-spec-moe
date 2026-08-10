# Phase 97 P4 — synchronous live same-event recorder wiring

Status: **PASS for synchronous engine wiring and CPU contract integration;
BLOCKED for w512, GPU measurement, P4a, admission, and performance claims.**

Date: 2026-08-09.

## Decision

The synchronous scheduler boundary now emits the frozen explicit target-step
event shape. This clears the implementation item previously named
`same_event_recorder_unwired` in the current readiness state. The frozen
runner/scorer JSON remains an unchanged pre-wiring snapshot; this result is an
additive engineering artifact and does not rewrite the adapter, scorer, or
their hashes.

The recorder is deliberately limited to the executable B0 `OFF` and
target-matching `K4` actions. It rejects a w512 capture configuration until
the separate mask-equivalence artifact exists.

## Live boundary

For every synchronous scheduler event with target decode rows, the recorder
binds:

```text
scheduler action and request rows
  + model-runner action, shared-KV, true-slot, and weight evidence
  + post-stop committed token counts
  + one monotonic scheduler-event interval
  -> one p4-same-event-target-step-v1 event
```

Every request row explicitly contains accepted draft tokens, raw generated
tokens, committed tokens, clipped tokens, and its armed state. Invalid
speculative-token counts are expanded to the same request-id set before the
event is built. No Prometheus delta or aggregate-to-request reconstruction is
used.

The opt-in interface is:

```text
VLLM_SELF_SPEC_KOFF_RUNTIME=1
VLLM_SELF_SPEC_P4_CAPTURE_CONFIG=/absolute/path/to/cell.json
VLLM_SELF_SPEC_P4_CAPTURE_OUTPUT=/absolute/path/to/capture.json
```

The config is a complete frozen capture object with `complete=false` and
`events=[]`. It supplies the registered cell metadata. The recorder replaces
the placeholder shared-KV binding, true-slot mapping, and draft-weight version
with live worker evidence. The output path is reserved create-new and replaced
atomically only by its owning recorder.

`complete=true` is emitted only after every one of the 32 configured request
ids has exactly `max_output_tokens` committed tokens. Shutdown before equal
work emits an incomplete, adapter-rejected capture when at least one event was
observed.

## Fail-closed behavior

The implementation rejects:

- missing config/output path pairs, asynchronous scheduling, or capture
  without the strict K/OFF runtime;
- an existing output, malformed nested capture fields, non-B0 actions, or a
  cell without exactly 32 unique request ids;
- missing per-request observations, unexpected request ids, duplicate or
  non-monotonic events, token-counter closure failures, and excess work; and
- any change in shared-KV binding/pool identity, true-slot identity, or
  target-matching weight version.

Mixed prefill/decode, preemption, recomputation, and invalid-spec events retain
their explicit quality flags and are ineligible. Action transitions retain
both verified and next action ids, so the unchanged adapter rejects the whole
round rather than silently treating them as steady state.

## Adapter proof

The new CPU integration constructs a complete 32-request R6/K4 capture solely
through the engine-side event builder and recorder. Its 52 events commit the
exact 8,192 requested tokens. The unchanged frozen adapter accepts the output,
including its explicit request rows and live proof ids.

This is live-format CPU contract evidence, not a GPU capture, warmup result,
or performance measurement.

## Artifacts

```text
vllm/envs.py
vllm/v1/spec_decode/koff_runtime.py
vllm/v1/core/sched/scheduler.py
tests/v1/spec_decode/test_koff_runtime.py
tests/v1/core/test_scheduler.py
research/97_composition_runtime/tests/test_p4_live_recorder.py
```

## Validation

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  tests/v1/spec_decode/test_koff_runtime.py \
  tests/v1/core/test_scheduler.py -q

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  research/97_composition_runtime/tests -q
```

The two vLLM files pass 149 CPU tests. The complete Phase 97 suite passes 178
tests plus 14 subtests. Ruff check and format-check pass all six touched Python
files.

No GPU command was run. Every GPU, P4a, admission, and production-value
authorization remains false.

## Next step

Prove that the boot-static w512 surrogate has acceptance-equivalent mask,
shared-KV, and true-slot semantics. Then produce the conservative candidate
resource bound over measured B0. Only a separate approval after both artifacts
exist may authorize the matched GPU screen.
