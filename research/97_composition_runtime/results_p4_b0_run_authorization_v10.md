# Phase 97 P4 — bounded chunked-prefill value-screen authorization V10

Date: 2026-08-10

Status: **CPU PASS; one source-bound GPU-4 V10 value-screen attempt is
AUTHORIZED at an absent create-only output and remains unexecuted**

## Outcome

V10 replaces the rejected V9 full-prefill geometry with the bounded
chunked-prefill cohort path. The consumed V5 GPU probe proved exactly the
target-matching K4 action for R4, R5, and R5cot at `8192 / 8160 / 0.90`; it did
not GPU-probe OFF, w512, or the full capture matrix. All-action and prompt-slice
rollover coverage is CPU-proven. V10 preserves the frozen B0 matrix, actions,
prompts, decode-work currency, adapter, and scorer. The `114688 / 0.96`
full-microbatch geometry is explicitly forbidden.

The package binds:

- the consumed V9 authorization and its immutable eight-capture R5 OOM;
- the V5 authorization, one-boot preparation, 40-record trace, and formal
  K4-only R4/R5/R5cot three-cohort non-scored PASS;
- the current matrix runner, recorder, scheduler, K/OFF runtime, model path,
  adapter, scorer, cohort proof, validators, and execution-path tests; and
- the fresh create-only `run_b0_value_screen_v9` output.

No GPU command was executed during this authorization review.

## Authorized boundary

```text
physical GPU / UUID                         4 / GPU-c9d19019-...
engine geometry                                   8192 / 8160 / 0.90
physical boots / captures                                  9 / 432
planned measurement cohorts                                  3,384
pre-V10 GPU probe coverage                         K4 / R4,R5,R5cot
all-action and prompt-slice rollover proof                       CPU
full-microbatch prefill                                      forbidden
capture-cohort barrier                                        required
prior-output reuse / partial resume / retry             false / false / false
fallback GPU                                                   false
value-screen scoring                         only after 432 complete captures
score grants authority                                         false
P4a / action admission / production claim        false / false / false
```

Every prompt slice carries the exact measurement-only cohort marker. All
members are queued before the first engine step, early prefill completers are
held, and the first measured decode is released atomically. A missing member,
abort, resource-floor violation, source drift, or incomplete capture stops the
attempt without scoring and requires a fresh authorization.

## Validation

```text
V10 authorization tests                              13 passed, 2 subtests
active runner/cohort/V10 surface                     59 passed, 7 subtests
complete active Phase 97 surface                   451 passed, 43 subtests
K/OFF runtime                                                  42 passed
complete scheduler                                             124 passed
Ruff check / format                                                  pass
V10 source-bound validator                                           pass
temporary prepare-only matrix                                        pass
V10 output                                                         absent
GPU execution in this task                                           none
```

The full historical Phase 97 audit produced 468 passes and 45 subtests. Its 21
failures are consumed-state assertions: 19 V5 pre-run tests correctly reject
the now-existing V5 output, and two V9 run-ready tests correctly reject the
runner's V10 source drift. No active V10 test failed.

The temporary prepare-only run materialized nine boot specs, 432 capture
cells, and 3,384 cohort slices with no capture file or GPU claim. Its temporary
directory was removed after inspection; the authorized create-only output was
not touched.

## Authorized command

The following command is registered but was not executed in this task:

```bash
.venv/bin/python \
  research/97_composition_runtime/scripts/run_p4_b0_value_screen.py \
  --authorization \
  research/97_composition_runtime/data/p4/p4_b0_run_authorization_v10.json \
  --output-dir \
  research/97_composition_runtime/data/p4/run_b0_value_screen_v9
```

This authority permits exactly one parent launch, up to nine sequential
physical boots on GPU 4, and scoring only after all 432 captures complete.
Any failure consumes V10 and stops without retry, resume, fallback, partial
reuse, or score.

## Artifacts

- authorization:
  `data/p4/p4_b0_run_authorization_v10.json`
  (`d772e983718cd8e4ccb16c506da1a7a0ef15908221b59c0ea98b014b5d37bc65`);
- validation:
  `data/p4/p4_b0_run_authorization_v10_validation.json`
  (`dc46685d0775ca4b7a4075a42bf85f4e856094cf7c4acabb100f3a9f4fc8baae`);
- schema: `schemas/p4_b0_run_authorization_v10.schema.json`;
- validator: `scripts/validate_p4_b0_run_authorization_v10.py`;
- focused tests: `tests/test_p4_b0_run_authorization_v10.py`; and
- launch runner: `scripts/run_p4_b0_value_screen.py`.
