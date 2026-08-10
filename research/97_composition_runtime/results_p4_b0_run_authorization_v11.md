# Phase 97 P4 — repaired variable-prefill value-screen authorization V11

Date: 2026-08-10

Status: **CPU PASS; one source-bound GPU-4 V11 value-screen attempt is
AUTHORIZED at an absent create-only output and remains unexecuted**

## Outcome

V11 preserves the consumed V10 attempt without reusing any of its 72 complete
captures or its empty R8 placeholder. It separately binds the immutable
GPU-0/GPU-1 repair audit. Both the isolated R8 case and the exact
R5cot-tail-to-R8 case completed without a runtime or shutdown exception and
contained exactly one truthful repaired R8 signature:

```text
draft_step0_query_width                                null
draft_step0_num_tokens                                 2116
draft_step0_batch_size                                   16
draft output shape                                  [16, 4]
produced draft width                                       4
```

The transition case's other eight armed-prefill observations are valid
chunked R5cot work. V11 therefore binds the parent rejection as an
observer-cardinality bug, not a GPU-case failure. The consumed repair package
itself granted no retry authority; this separate source-bound review grants
one new attempt.

The bounded `8192 / 8160 / 0.90` cohort-barrier geometry, frozen prompts,
actions, decode-work currency, adapter, and scorer are unchanged. The active
cohort CPU proof was revalidated against the repaired K/OFF runtime source.
The `114688 / 0.96` full-prefill geometry remains forbidden.

## Authorized boundary

```text
physical GPU / UUID                         4 / GPU-c9d19019-...
engine geometry                                   8192 / 8160 / 0.90
physical boots / captures                                  9 / 432
planned measurement cohorts                                  3,384
fresh output                                  run_b0_value_screen_v10
V10 capture reuse / repair-output reuse                 false / false
partial resume / retry / fallback GPU             false / false / false
value-screen scoring                         only after 432 complete captures
score grants authority                                         false
P4a / action admission / production claim        false / false / false
```

Any source or evidence drift, existing output, resource-floor failure,
incomplete cohort, abort, or matrix drift consumes the attempt and stops it
without scoring. The runner accepts only the exact reviewed authorization and
output pair in both parent and child dispatch.

## Validation

```text
V11 authorization tests                              14 passed, 4 subtests
active runner/runtime/source-bound surface          251 passed, 15 subtests
consumed repair pre-run assertion                                deselected
refreshed cohort-barrier proof                       19 passed, 2 subtests
Ruff check / format                                                  pass
V11 source-bound validator                                           pass
temporary prepare-only matrix                        9 boots / 432 captures
authorized V11 output                                             absent
GPU execution in this task                                           none
```

The deselected repair assertion requires its create-only output to be absent;
that output now exists because the package was consumed. Its immutable
case-level preservation tests are included in the passing surface.

The temporary preparation used an automatically removed directory. It wrote
nine boot specs and nine plans, closed to 432 cells, retained only the bounded
chunked-prefill budget, and did not touch the authorized output.

## Authorized command

The following command is registered but was not executed in this task:

```bash
.venv/bin/python \
  research/97_composition_runtime/scripts/run_p4_b0_value_screen.py \
  --authorization \
  research/97_composition_runtime/data/p4/p4_b0_run_authorization_v11.json \
  --output-dir \
  research/97_composition_runtime/data/p4/run_b0_value_screen_v10
```

It permits one parent launch and up to nine sequential physical boots on GPU
4. Scoring is permitted only after all 432 captures complete. A failure does
not permit resume, retry, fallback, partial reuse, or scoring.

## Artifacts

- authorization: `data/p4/p4_b0_run_authorization_v11.json`
  (`143c831383d9e7b2f9b1a4f020c8bbf8c185152f9c94537fb2d2c657eb72f60f`);
- validation: `data/p4/p4_b0_run_authorization_v11_validation.json`
  (`1aaf9ae0470a855e63bafd4bf6f6f7cff625b0f531cf5743fa02ce8453da8196`);
- schema: `schemas/p4_b0_run_authorization_v11.schema.json`;
- validator: `scripts/validate_p4_b0_run_authorization_v11.py`;
- focused tests: `tests/test_p4_b0_run_authorization_v11.py`; and
- launch runner: `scripts/run_p4_b0_value_screen.py`.
