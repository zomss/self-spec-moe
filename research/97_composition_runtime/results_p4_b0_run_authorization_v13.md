# P4 B0 value-screen run authorization V13

Date: 2026-08-10

Decision: **APPROVE one block-parallel, two-lane, nine-boot / 432-capture
matched B0 value screen.** V13 is V12 with the launcher path repair. The
lane assignment, decision boundary, scoring rule, and every downstream
prohibition are unchanged: P4a engineering, action admission, and any
performance claim remain false.

## What consumed V12

V12's one exact invocation refused before any GPU work. It passed the
worktree-quiescence, native-sampler, in-process EngineCore, and both GPU
identity/idle preflights, and it prepared the nine-boot plan package. It then
raised `ValueError` inside `snapshot_sources` because
`Path.relative_to(REPO_ROOT)` was applied to a snapshot target built from a
**relative** `--output-dir`.

The attempt loaded no model, launched no boot child, and emitted zero
captures. The immutable record is
`data/p4/run_b0_value_screen_v11/failure.json`, classified
`launcher_relative_output_path_not_normalized` with
`runtime_fault_observed: false` and `gpu_fault_observed: false`. V12 is
consumed because its registered create-only output directory now exists; that
output is preserved and never reused.

This is the same defect class as
`results_p4_b0_chunked_prefill_probe_attempt_v1.md`, which was also consumed
by an unnormalized relative output path. Repeating a known class is the part
worth naming plainly.

## Repair

- `snapshot_sources` and `verify_against_snapshot` resolve the output
  directory before building any path;
- `execute_run` resolves both the authorization and output paths, and derives
  the registered output through the shared reviewed-pair dispatcher rather
  than a launcher-local constant;
- `main` resolves both paths once, before any use; and
- snapshot copies are now hashed **directly** rather than round-tripped
  through a repository-relative path, so the verification no longer depends on
  where the run directory sits.

Two regression tests drive the launcher with a relative `--output-dir` and
assert that snapshot creation and verification agree. Both fail against the
pre-repair launcher.

## What is unchanged from V12

The lane assignment (lane-a = GPU 0 with blocks 1 and 3; lane-b = GPU 1 with
block 2), the never-split-a-block rule, the Phase 96 read-only precedent and
its 0.45%/0.49% measured spread, the 2% cross-boot certification tripwire, the
source snapshot and executing-code guard, the worktree-quiescence preflight,
and the capture block as restart unit. See
`results_p4_b0_run_authorization_v12.md` for the full derivation of why a
lane-wide scale factor cancels in the scored ratio.

## Verification

- V13 validates: `data/p4/p4_b0_run_authorization_v13_validation.json`,
  status `pass`, two lanes, nine boots, 432 captures, fresh output ready.
- Phase CPU suite: 38 failures, byte-identical to the pre-existing baseline;
  zero regressions. The V12 package test now asserts its consumed refusal
  instead of validation, which is what a consumed package must do.

## Artifacts

- `data/p4/p4_b0_run_authorization_v13.json`
- `data/p4/p4_b0_run_authorization_v13_validation.json`
- `data/p4/run_b0_value_screen_v11/failure.json` (consumed V12 attempt)
- `schemas/p4_b0_run_authorization_v13.schema.json`
- `scripts/validate_p4_b0_run_authorization_v13.py`
- `tests/test_p4_b0_run_authorization_v13.py`

Registered create-only output: `data/p4/run_b0_value_screen_v12`.
