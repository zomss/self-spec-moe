# Phase 97 P4 — B0 value-screen attempt V2

Status: **STOPPED before the first capture. V3 is consumed, this attempt is
invalid for scoring, and its output cannot be resumed or retried. GPU 5, P4a,
action admission, and performance claims remain unauthorized.**

Date: 2026-08-09.

## Result

The exact V3 invocation passed the repaired virtualenv/Ninja preflight and
reached the first GPU-4 engine boot. Target and draft compilation completed,
self-spec weight sharing reported 291 aliased parameters, and all 36 draft
attention layers bound to target-owned KV without a private draft allocation.

Engine initialization then stopped while loading FlashInfer's cached sampling
module during the sampler profile. The pre-existing `sampling.so` requires
`libcudart.so.13`, but that soname was not resolvable through the child
process's dynamic-library search path. TVM FFI consequently rejected the
module before the first capture.

```text
failed boot                          p4-b0-b1-p1-off
completed boots                                     0
captures                                            0
adapted rounds                                      0
score emitted                                   false
Ninja preflight                                  pass
root exception   sampling.so: libcudart.so.13 not found
```

This is an execution-environment failure, not a value, acceptance, KV-capacity,
or performance result. The environment evidence is:

- PyTorch `2.11.0+cu129` reports CUDA `12.9`;
- `/usr/local/cuda` resolves to the CUDA 13.0 toolkit;
- the cached FlashInfer 0.6.12 module has a `NEEDED` entry for
  `libcudart.so.13`; and
- the child had no `LD_LIBRARY_PATH` entry resolving that dependency.

## Fail-closed disposition

The parent exited on the first non-zero child result. No retry, partial resume,
fallback GPU, adaptation, or scoring occurred. No engine or GPU compute process
remained after failure. The create-new V2 output is preserved as a consumed,
failed attempt and must never be deleted or reused.

The machine-readable record is
`data/p4/run_b0_value_screen_v2/failure.json`.

## Required repair

Before another authorization, the launcher must:

1. choose and bind an explicit CUDA runtime/toolchain compatible with the
   FlashInfer sampling module and the installed PyTorch build;
2. preflight loading the actual sampling module, not only discovering Ninja,
   before creating the output directory;
3. cover dependency-resolution failure in CPU/environment tests; and
4. produce a new source-bound authorization with another create-only output
   path.

V3 is consumed and cannot authorize another GPU command.
