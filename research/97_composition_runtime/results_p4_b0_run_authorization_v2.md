# Phase 97 P4 — source-bound B0 value-screen authorization V2

Status: **APPROVE for one exact GPU-4 boot-static OFF/K4/w512 value screen.
The package is executable and source-current. No GPU workload has run yet.
P4a, runtime w512 switching, action admission, and performance claims remain
unauthorized.**

Date: 2026-08-09.

## Decision

The historical V1 HOLD remains immutable and non-executable. The additive V2
review validates that its six blockers are cleared, binds every current
conformance source, and authorizes only this measurement scope:

```text
physical GPU                         4 only
physical boots                       9
cells per boot                      48
required complete captures         432
shared-KV launch/capture floor   21,682 blocks
fallback GPU                         forbidden
partial resume                       forbidden
P4a or admission authority           false
```

The authorization is for a boot-static acceptance/value screen. The w512
candidate receives no latency or cost credit and is not admitted as a runtime
action.

## Source and contract closure

The package binds the conformance artifact plus its exact historical HOLD,
runtime, scheduler, model runner, environment registry, matrix runner, tests,
schema, and validator hashes. The live runner rechecks every hash before an
engine can start.

It also compares the approved package against the frozen V1 model, GPU,
engine, environment, action boots, matrix, and resource gates. Only the
invocation changes from held to launchable. The runner rejects:

- another package id, authorization path, or output path;
- source, model, GPU, engine, action, matrix, or resource drift;
- an incomplete or inflated source closure;
- an existing output directory or overwrite attempt;
- GPU 5 fallback or partial-matrix resume; and
- P4a, admission, or production-value authority.

Any run failure stops without scoring and requires a fresh authorization. The
scorer cannot grant downstream authority.

## Exact authorized invocation

```bash
.venv/bin/python \
  research/97_composition_runtime/scripts/run_p4_b0_value_screen.py \
  --authorization \
  research/97_composition_runtime/data/p4/p4_b0_run_authorization_v2.json \
  --output-dir \
  research/97_composition_runtime/data/p4/run_b0_value_screen_v1
```

No other GPU invocation is authorized by this package.

## Artifacts

```text
schemas/p4_b0_run_authorization_v2.schema.json
data/p4/p4_b0_run_authorization_v2.json
data/p4/p4_b0_run_authorization_v2_validation.json
scripts/validate_p4_b0_run_authorization_v2.py
tests/test_p4_b0_run_authorization_v2.py
scripts/run_p4_b0_value_screen.py
```

## Validation

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research/97_composition_runtime/scripts/validate_p4_b0_run_authorization_v2.py \
  --authorization \
  research/97_composition_runtime/data/p4/p4_b0_run_authorization_v2.json

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  research/97_composition_runtime/tests/test_p4_b0_run_authorization_v2.py -q

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover \
  -s research/97_composition_runtime/tests -p 'test_*.py' -v
```

The focused V2 suite passes 12 tests. The complete Phase 97 suite passes 253
tests plus 21 subtests. Ruff 0.14.0 check passes the runner, validator, and V2
tests. The validator reports a current source closure, nine boots, 432
captures, GPU 4 only, and a create-new-ready output path. No GPU workload was
run while producing this review.

## Next step

Execute the exact authorized invocation and produce
`p4_b0_value_screen_result`. The result must contain all 432 captures before
adaptation and scoring. A dominance stop keeps K4/OFF if w512 does not improve
acceptance; any positive value result still requires separate P4a and
post-capture resource reviews.
