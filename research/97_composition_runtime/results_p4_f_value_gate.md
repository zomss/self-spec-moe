# Phase 97 P4 entry — Phase 96/F value decision

Status: **NOT APPROVED. The value of
`target-matching-w512-masked-k4` is unresolved, not disproven. P4a
engineering, GPU measurement, live admission, and performance claims remain
unauthorized.**

Date: 2026-08-09.

## Decision

The Phase 96/F pre-engineering gate cannot approve the P4 masked-window
action. Validation of the decision package passes, but the approval state is
`not_approved`; those are intentionally separate fields.

| F prerequisite | observed state | gate |
| --- | --- | --- |
| scored workload objective and declared `w_g` | workload is an unscored `engineering_assumption`; objective is capacity only | fail |
| exact action value | no target-matching masked-w512 acceptance or cost result | fail |
| Phase 96/D result | 21 complete files exist, but no scored result exists and accounting is invalid | fail |
| Phase 96/E result | absent | fail |
| global robust portfolio and sensitivity | absent; gain fields remain `null` | fail |
| exact P4 resource evidence | optimistic proxy; planner decision is `reject` | fail |
| switch/probe overhead | unmeasured | fail |

Supporting W6 and W14/B evidence still motivates `w512`, but it used a
quantized draft or a separately booted realization. It cannot supply the
action-specific acceptance, same-realization cost, or resource-adjusted value
required for this target-matching masked action.

## W14/D audit

The audit finds the registered 21 files and all are marked complete. Their
bundle contains 1,680 rounds and has SHA-256
`911409a1a103fabd62bfb5fda672a69c6b1bad6bf3c4a986193edae57e192ee3`.
That is data presence, not a scored D result:

- `results_w14d.md` is absent;
- the frozen scorer reports the number of files but does not produce the D
  coverage, error, or false-elimination decision; and
- 313/1,680 rounds violate the binding relationship
  `U_unarmed = H_target_steps - D_armed >= 0`, with a minimum of -8.

The stored `closure_ok` flag does not resolve the last defect. The runner
derives `H` from `E + C - A`, so `E + C = A + H` closes by construction; it
does not prove that armed request-steps are a subset of target request-steps.
The D bundle must not feed F until the counter-boundary semantics are repaired
and the affected measurements are either justified under a newly registered
contract or recollected.

## Fail-closed interpretation

Missing measurements are represented as `null`, never as zero. Therefore this
decision does not say that the window has zero value. It says there is no
valid lower confidence bound that can be compared with the W3 threshold and
no weighted portfolio objective that can select more than the current K4/OFF
pool.

The checked authorizations are all false:

- no P4a scheduler/proposer engineering;
- no GPU run under this decision;
- no addition to the executable action registry; and
- no performance or service-value claim.

## Next artifact

Prepare a matched target-matching B0 value-screen preregistration, not an
implementation patch. Before it can authorize a run, it must:

1. freeze a scored service objective and workload weights;
2. repair target-step accounting and state the disposition of the current D
   bundle;
3. register matched `OFF`, `target-matching-k4`, and
   `target-matching-w512-masked-k4` controls; and
4. predeclare how action-specific acceptance, cost, W3 robust-value
   thresholds, resource estimates, and later switch/probe overhead determine
   engineering approval.

If that screen clears the robust pre-engineering threshold, P4a can receive a
new explicit authorization. Exact post-engineering capacity, live identity,
target-local correctness, and transition overhead would still be required
before admission.

Follow-up: that preregistration is now present in
`results_p4_b0_value_screen_preregistration.md`. It freezes a research-only
equal-weight W3 objective and same-event accounting, but remains blocked on
five run-readiness artifacts. It does not change this `not_approved` decision
or authorize GPU work or P4a.

## Artifacts and commands

```text
schemas/p4_f_value.schema.json
data/p4/p4_f_value_decision.json
data/p4/p4_f_value_validation.json
scripts/validate_p4_f_value.py
tests/test_p4_f_value.py
```

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research/97_composition_runtime/scripts/validate_p4_f_value.py \
  --decision \
  research/97_composition_runtime/data/p4/p4_f_value_decision.json

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research.97_composition_runtime.tests.test_p4_f_value -v
```

The focused F suite passes 19 tests, the full Phase 97 CPU suite passes 86
tests, and Ruff 0.14.0 passes the F validator and tests.
