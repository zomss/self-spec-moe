# Phase 97 P4 — B0 capture-runner conformance

Status: **PASS for all six registered CPU conformance checks. The executable
source is ready for a fresh authorization review, but the historical HOLD
package cannot launch it. GPU measurement, P4a, admission, and performance
claims remain unauthorized.**

Date: 2026-08-09.

## Decision

The narrow engineering authorized by the prior review is complete:

| conformance check | result |
| --- | --- |
| boot-static w512 contract | pass: exact 512-token window and 16 sinks |
| recorder w512 acceptance | pass: only under the validated boot contract |
| multi-cell same-boot capture | pass: 48 ordered create-new cells per boot |
| trusted boot-static relabel | pass: exact live K4-to-K4 events only |
| logical weight identity | pass: stable cross-boot ID plus live pointer proof |
| runtime resource-floor guard | pass: 21,682 blocks at launch and capture |

This is executable-conformance evidence, not value or performance evidence.
No GPU command was run.

## Live contracts

The normal executable action registry remains the P3 `OFF`/K4 pair. The w512
surrogate is admitted only as a boot-static capture label when all P4 capture
fields are present and agree with `window=512`, `sinks=16`, synchronous
scheduling, shared target KV, and K4. The recorder relabels only an exact live
K4-to-K4 event with K=4 and consistently armed request rows; it cannot turn an
OFF or transition event into w512 evidence.

The runner derives one logical draft-weight identifier from the model id,
revision, target quantization, and target-matching draft path. That identifier
is stable across all nine processes. It supplements rather than replaces the
live within-boot checks: target and draft parameters must still be the same
objects with identical storage pointers, and each event carries the resulting
binding id.

The scheduler checks the target-owned shared-KV pool at recorder construction
and immediately before every capture. A capacity below 21,682 blocks stops the
run without scoring. Pool identity, shared aliases, true-slot identity,
preemption, recomputation, and equal-work closure remain fail-closed.

## Matrix runner

`run_p4_b0_value_screen.py` constructs the frozen counterbalanced matrix:

```text
3 boot blocks x 3 actions                    =   9 physical boots
6 regimes x 2 content seeds x 4 rounds      =  48 cells per boot
9 physical boots x 48 cells                  = 432 captures
```

One recorder rotates through all 48 canonical `regime, seed, round` cells in a
single engine process. Outputs use create-new semantics, and missing, repeated,
or out-of-order cells fail. `--prepare-only` materializes and validates the
nine boot plans without constructing an engine or granting GPU authority.

Execution is deliberately source-bound. The checked historical package has a
HOLD decision and pre-conformance hashes, so both the validator and runner
reject it. A future package must bind the checked conformance artifact, mark
all six checks as passing, and explicitly authorize GPU measurement before a
child process can start. Scoring never grants authority.

## Artifacts

```text
vllm/v1/spec_decode/koff_runtime.py
vllm/v1/core/sched/scheduler.py
vllm/v1/worker/gpu_model_runner.py
vllm/envs.py
research/97_composition_runtime/scripts/run_p4_b0_value_screen.py
research/97_composition_runtime/schemas/p4_b0_capture_runner_conformance.schema.json
research/97_composition_runtime/data/p4/p4_b0_capture_runner_conformance.json
research/97_composition_runtime/scripts/validate_p4_b0_capture_runner_conformance.py
research/97_composition_runtime/tests/test_p4_b0_capture_runner.py
research/97_composition_runtime/tests/test_p4_b0_capture_runner_conformance.py
```

## Validation

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  tests/v1/spec_decode/test_koff_runtime.py \
  tests/v1/core/test_scheduler.py \
  research/97_composition_runtime/tests/test_p4_live_recorder.py \
  research/97_composition_runtime/tests/test_p4_b0_capture_runner.py \
  research/97_composition_runtime/tests/test_p4_b0_capture_runner_conformance.py \
  -k 'koff or p4' -q

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover \
  -s research/97_composition_runtime/tests -p 'test_*.py' -v

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research/97_composition_runtime/scripts/validate_p4_b0_capture_runner_conformance.py \
  --conformance \
  research/97_composition_runtime/data/p4/p4_b0_capture_runner_conformance.json
```

The focused suite passes 52 tests with 113 deselected. The complete Phase 97
suite passes 241 tests plus 21 subtests. The conformance validator reports six
checks, nine boots, 48 cells per boot, 432 total captures, and zero GPU
commands. Ruff 0.14.0 check passes the scoped Python files.

## Next step

Produce a fresh P4 B0 run-authorization review bound to the changed source and
conformance hashes. That review must decide GPU authority explicitly; this
artifact does not. Even a later authorized value-screen pass cannot admit w512
without the registered post-capture resource and P4a gates.
