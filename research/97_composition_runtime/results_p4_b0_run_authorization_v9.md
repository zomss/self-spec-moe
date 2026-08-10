# Phase 97 P4 — B0 launch-dispatch authorization V9

Status: **HISTORICAL AUTHORIZATION — CONSUMED. V9 bound the tested
parent/child launch-dispatch repair to one GPU-4-only, nine-boot, 432-capture
value-screen attempt. It was executed exactly once, completed eight R4
captures, and stopped on CUDA OOM in the first R5 full-prefill. P4a, action
admission, retry, fallback, and performance claims remain unauthorized.**

Date: 2026-08-09.

## Failure closed by V9

The exact V8 command resolved and validated the V8 package, then failed closed
at the parent launch pathname gate. Package resolution and execution-authority
validation admitted V8, but the final parent and child dispatchers still
listed only V6 and V7. Exit code 2 occurred before sampler, EngineCore, GPU
identity, output creation, model execution, capture, adaptation, or scoring.

The preserved failure records zero complete captures, zero incomplete
captures, no GPU execution, no score, and no retry. V8 is consumed.

V9 replaces the duplicate parent and child branches with one exact-path
resolver. That resolver maps each reviewed authorization to only its paired
create-new output and rejects unknown paths, cross-paired outputs, and child
specs outside the reviewed output. CPU regression tests exercise both the
parent `execute_run` path and child `main` path. The V9 validator also resolves
the registered invocation through this shared gate, closing the validation
gap that allowed V8's false-positive launchability result.

## Bound evidence and authority

V9 binds the immutable V8 authorization and pre-GPU refusal, the shared
dispatcher and both execution-path tests, frozen V8 request-ID and decode-work
repairs, retained atomic-ingress and resource evidence, every current
execution source, GPU 4 UUID
`GPU-c9d19019-5065-2353-80a9-f1797eb19d51`, and a fresh output:

```text
output path                         run_b0_value_screen_v8
physical GPU                                           4
EngineCore                                  InprocClient
parent / child dispatcher                         tested
measured decode / prefill offset                 S_dec / 1
physical boots / captures                         9 / 432
max_num_batched_tokens                            114688
gpu_memory_utilization                              0.96
shared target KV                                required
private draft KV                                 forbidden
```

V9 permits no GPU fallback, retry, partial resume, prior-output reuse, P4a
engineering, action admission, or production-performance claim. Any failure
must stop without scoring and requires another fresh authorization.

## Pre-execution validation

The following checks passed during the authorization review, before the later
one-shot execution created the V9 output directory:

```text
V8/V9 authorization focused tests      21 passed, 4 subtests
runner + conformance focused tests      26 passed, 3 subtests
complete Phase 97 suite               416 passed, 41 subtests
vLLM K/OFF runtime tests                          38 passed
Ruff check / format                                  pass
V9 source-bound validator                            pass
GPU model execution                                  none
V9 output directory                                absent
```

The historical command that later consumed V9 was:

```bash
.venv/bin/python \
  research/97_composition_runtime/scripts/run_p4_b0_value_screen.py \
  --authorization \
  research/97_composition_runtime/data/p4/p4_b0_run_authorization_v9.json \
  --output-dir \
  research/97_composition_runtime/data/p4/run_b0_value_screen_v8
```

This command was not run during the V9 review; it was run once afterward under
the separate execution request. The resulting failure is recorded in
`results_p4_b0_value_screen_attempt_v8.md`.

## Artifacts

- V8 refusal: `data/p4/run_b0_value_screen_v7/failure.json`
  (`ffeb2039511b3cf2105d015a3b3aafb1fa6fb83b5c03ea62b7bef138d6dcb600`);
- authorization: `data/p4/p4_b0_run_authorization_v9.json`
  (`1542763ea55e6cda9aaf29fda357bc3818a508eecd34d22d869c22aa13ea1ee4`);
- validation: `data/p4/p4_b0_run_authorization_v9_validation.json`
  (`13d2ea40c55abbe4696f365e1f9427129a8d4bb3b40f708bdc175f0a55c8b77c`);
- schema: `schemas/p4_b0_run_authorization_v9.schema.json`;
- validator: `scripts/validate_p4_b0_run_authorization_v9.py`; and
- focused tests: `tests/test_p4_b0_run_authorization_v9.py`.

V9 cannot be executed again. The current gate is the CPU-only
chunked-prefill cohort-barrier design/proof identified in
`results_p4_b0_full_prefill_transient_bound.md`; no GPU probe or V10 is
authorized.
