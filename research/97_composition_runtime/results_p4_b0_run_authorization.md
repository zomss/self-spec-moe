# Phase 97 P4 — B0 value-screen run-authorization review

Status: **HOLD. Evidence readiness is complete, but the registered screen is
not executable with the current live engine. GPU measurement is not
authorized. Only narrowly scoped capture-runner conformance engineering is
authorized.**

Date: 2026-08-09.

## Decision

The separate authorization review is complete and fails closed:

| decision axis | result |
| --- | --- |
| frozen research objective, prompts, adapter, and scorer | ready |
| synchronous OFF/K4 same-event evidence path | ready |
| boot-static w512 acceptance equivalence | ready, no latency credit |
| conservative B0 resource evidence | ready: 21,682-block floor |
| executable nine-boot capture path | not ready |
| GPU measurement | not authorized |
| P4a, action admission, or value claim | not authorized |
| capture-runner conformance engineering | authorized |

Evidence readiness and executable readiness are different gates. The three
additive artifacts cleared the preregistered evidence gaps; they did not prove
that the current engine can produce the exact 432-record matrix accepted by
the frozen scorer.

## Registered held run

The package records the intended run so conformance work cannot silently
change the experiment:

- Qwen3-8B revision
  `b968826d9c46dd6066d109eabc6255188de91218`, target-matching aliased draft,
  BF16 target-owned shared KV, TP1/PP1;
- physical GPU 4 only, UUID
  `GPU-c9d19019-5065-2353-80a9-f1797eb19d51`, H100 80GB HBM3;
- max model length 20,480, max batched tokens 8,192, max sequences 32,
  synchronous scheduling, compiled execution, no prefix cache, and K4;
- three counterbalanced boot blocks, each containing separate OFF, K4, and
  boot-static w512 physical boots;
- 48 cells per physical boot: six regimes times two content seeds times four
  rounds; and
- nine physical boots and 432 raw captures/adapted rounds in total.

The OFF control uses `[[1,32,0],[33,33,4]]`. Batch 33 is unreachable at the
registered 32-sequence limit and retains the same resident K4 graph pool as
the speculative controls. K4 and w512 use `[[1,32,4]]`. The w512 realization
uses a 512-token window with 16 sinks and receives no window cost credit.

All nine boots must remain on GPU 4 because hardware identity is part of the
adapter's matched stable configuration. GPU 5 is recorded but unauthorized;
using it requires a new package and a complete matrix restart.

The intended invocation is registered as:

```bash
.venv/bin/python \
  research/97_composition_runtime/scripts/run_p4_b0_value_screen.py \
  --authorization \
  research/97_composition_runtime/data/p4/p4_b0_run_authorization.json \
  --output-dir \
  research/97_composition_runtime/data/p4/run_b0_value_screen_v1
```

That runner does not exist, so this command is a held interface, not an
executable authorization.

## Executable-conformance blockers

Six checks fail in the current source:

1. `w512_boot_contract_blocked`: `validate_boot_config()` requires
   `draft_kv_window == 0`, so it rejects the registered window-512 boot.
2. `w512_recorder_action_blocked`: `P4SameEventRecorder` permits only OFF and
   K4 capture templates and explicitly rejects the w512 action.
3. `multi_cell_same_boot_capture_missing`: the recorder owns one config/output
   cell, while the frozen scorer requires 48 cells without replacing each
   physical boot. The matrix runner is also absent.
4. `boot_static_action_relabel_missing`: same-event records inherit action ids
   from the two-action OFF/K4 registry. No fail-closed boot-static proof
   relabels K4 execution as the registered w512 acceptance realization.
5. `logical_weight_version_unstable`: live alias validation derives the draft
   weight version from storage pointers and overwrites the configured logical
   version. Those pointers change across processes, but the scorer requires
   one stable configuration hash across all nine boots.
6. `runtime_resource_floor_guard_missing`: events report shared-KV capacity,
   but launch/capture does not stop below the registered 21,682-block floor.

The fixes must preserve the existing within-boot pointer/alias proofs while
adding a cross-boot logical weight-version identifier. The w512 label must be
derived only from a validated boot-static window contract; it cannot be a
free-form capture-template claim. Multi-cell capture must rotate create-new
cell outputs inside one physical engine boot and reject missing, repeated, or
out-of-order matrix cells.

## Resource stop condition

The authorization retains the conservative B0 arithmetic:

```text
base shared-KV capacity                    24,529 blocks
conservative debit                          2,847 blocks
minimum launch/capture floor               21,682 blocks
required workload envelope                 21,000 blocks
conservative headroom                         682 blocks
```

The conformance runner must stop without scoring if capacity is below 21,682,
the target-owned pool count is not one, live alias/slot proofs fail, or any
preemption or recomputation enters a scored cell.

## Artifacts and validation

```text
schemas/p4_b0_run_authorization.schema.json
data/p4/p4_b0_run_authorization.json
data/p4/p4_b0_run_authorization_validation.json
scripts/validate_p4_b0_run_authorization.py
tests/test_p4_b0_run_authorization.py
```

Validate with:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research/97_composition_runtime/scripts/validate_p4_b0_run_authorization.py \
  --authorization \
  research/97_composition_runtime/data/p4/p4_b0_run_authorization.json

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  research/97_composition_runtime/tests/test_p4_b0_run_authorization.py -q
```

The focused suite passes 17 tests. It covers the source bindings, exact held
invocation, GPU and model identity, matrix closure, resource floor, all six
blockers, and authority inflation. The complete Phase 97 CPU suite passes 228
tests plus 21 subtests. Ruff 0.14.0 check and format-check pass the two new
Python files. No GPU command was run.

## Next step

Produce `p4_b0_capture_runner_conformance`: implement and CPU-test the narrow
boot-static w512 contract, trusted action labeling, stable logical weight
version, multi-cell same-boot recorder/runner, and resource-floor stop. That
artifact may not self-authorize a GPU run. After all six checks pass, perform
a new explicit authorization review against the changed source hashes.
