# Phase 97 P4 — B0 value-screen attempt V1

Status: **STOPPED before the first capture. The attempt is invalid for scoring
and cannot be resumed or retried under V2. GPU 5, P4a, action admission, and
performance claims remain unauthorized.**

Date: 2026-08-09.

## Result

The exact V2 invocation reached the first GPU-4 engine boot. Target and draft
compilation completed, self-spec weight sharing reported 291 aliased
parameters, and all 36 draft attention layers bound to the target KV cache
without draft-side KV allocation.

Engine initialization then stopped during its sampler profiling run. FlashInfer
attempted to JIT-build its sampling module by invoking `ninja`, but the direct
`.venv/bin/python` launch did not add `.venv/bin` to `PATH`. The Ninja package
and executable were already installed; executable discovery was the missing
launcher precondition.

```text
failed boot                  p4-b0-b1-p1-off
completed boots                              0
captures                                     0
adapted rounds                               0
score emitted                            false
root exception   FileNotFoundError: 'ninja'
```

This is an execution-environment failure, not a value, acceptance, KV-capacity,
or performance result.

## Fail-closed disposition

The parent exited on the first non-zero child result. No retry, partial resume,
fallback GPU, adaptation, or scoring occurred. No engine or GPU compute process
remained after failure. The create-new V1 output is preserved as a consumed,
failed attempt and must never be deleted or reused.

The machine-readable record is
`data/p4/run_b0_value_screen_v1/failure.json`.

## Required repair

Before creating another authorization, the runner must:

1. derive the executable directory from the active virtual environment;
2. prepend that directory to each boot child's `PATH`;
3. fail before output creation unless `ninja` resolves to an executable and
   responds successfully; and
4. prove those conditions in CPU-only tests.

Any new attempt requires a new source-bound package and a new create-only
output directory. V2 is consumed and cannot authorize another GPU command.
