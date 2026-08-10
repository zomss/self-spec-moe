# Phase 97 P4 — variable-prefill evidence repair validation

Status: **GPU CASE-LEVEL PASS; PARENT AGGREGATE REJECTED BY ITS OBSERVER.**
The isolated R8 and exact R5cot-tail-to-R8 workloads both completed without a
runtime or shutdown exception. The source-bound parent then rejected its own
aggregate because it expected two armed-prefill observations in the transition
case, while real chunked R5cot produced eight before the one R8 observation.
The attempt is consumed, immutable, non-scored, and grants no V11 or value-screen
authority.

Date: 2026-08-10.

## Runtime repair

`draft_step0_query_width` remains a nullable exact uniform-width fact. The live
proposer and runner now also transport:

- `draft_step0_num_tokens`; and
- `draft_step0_batch_size`.

K4 evidence requires positive work values, token count at least batch size,
draft output rows equal to batch size, output width equal to K, a called
proposal, and both execution modes. If a uniform width is present, its product
with batch size must equal token count. Pure decode still requires exact width
one. Armed pure prefill may retain a null width when its request widths differ;
it does not manufacture an average.

## CPU closure

The exact R8 regression uses the diagnosed geometry: 2,100 scheduled target
tokens, 16 appended sampled tokens, 2,116 draft step-0 tokens, batch 16,
maximum query width 344, and output shape `[16, 4]`. It runs with both no prior
width and a stale prior width-one value to cover isolated R8 and R5cot-to-R8.

Fail-closed tests reject missing work facts, mismatched output rows, fabricated
uniform width, nullable pure-decode width, and unarmed variable prefill.

```text
tests/v1/spec_decode/test_koff_runtime.py + tests/v1/core/test_scheduler.py
  171 passed

tests/v1/spec_decode/test_koff_runtime.py + test_p4_live_recorder.py
  56 passed, 6 subtests passed

repair authorization/package tests
  7 passed

immutable preservation/audit tests
  5 passed
```

Ruff passes for the new validation and preservation files. A whole-file Ruff
check on the already-dirty proposer still reports two pre-existing findings at
lines 2446 and 2874; neither is in this repair.

## GPU outcome

Both jobs retained Qwen3-8B revision
`b968826d9c46dd6066d109eabc6255188de91218`, target-matching K4, one
target-owned 36-layer KV cache, 291 target/draft weight aliases, and
`8192 / 8160 / 0.90` chunked-prefill geometry.

| Case | GPU | Completed cohorts | Measured events | Trace records | Result |
|---|---:|---|---:|---:|---|
| isolated R8 | 0 | R8 | 610 | 612 | pass |
| R5cot → R8 | 1 | exact R5cot tail, then R8 | 1,273 | 1,289 | pass |

Each case had 24,527 shared target-KV blocks against the 21,682-block floor,
closed its recorder, completed its full output work, and released its GPU.

The exact R8 arm appeared once in each case:

```text
capture_cohort_arm                                      true
pure_decode                                            false
decode_req_ids                                            []
proposal_called                                        true
draft_step0_query_width                                null
draft_step0_num_tokens                                 2116
draft_step0_batch_size                                   16
draft output shape                                  [16, 4]
draft step-0 runtime mode                              NONE
draft chain runtime mode                          PIECEWISE
validated produced draft width                            4
```

This is direct GPU evidence that the repaired schema accepts truthful
variable-width prefill while the subsequent decode path continues through the
width-one compaction.

## Parent aggregate rejection

The transition case recorded nine armed-prefill observations:

- eight valid chunked R5cot prefill arms; and
- one exact R8 arm at engine step 632.

The parent validator incorrectly required exactly two observations, treating
one logical cohort as one prefill arm. It therefore exited with:

```text
r5cot-to-r8 armed-prefill evidence count drifted
```

This occurred after both case results and the base aggregate diagnosis were
written. The base classification is `original_failure_not_reproduced`.
The external immutable audit locates exactly one R8 observation by its complete
signature rather than counting unrelated prefill arms. It proves the parent
rejection caused no GPU-case failure, but it does not retroactively turn the
consumed package into a passing authorization.

## Immutable artifacts

- authorization: `p4_b0_variable_prefill_repair_validation_authorization_v1.json`
  (`bb493e0e11ffd0f37e08c8cb30875c282cf4360979391513c65b5b1ed16af961`);
- base aggregate: `run_b0_variable_prefill_repair_validation_v1/diagnosis.json`
  (`55cd0e5b6f22b01235cee8ccb28ac5a9ae2ada6a740fdcd7080cd209104382b3`);
- isolated case result:
  `run_b0_variable_prefill_repair_validation_v1/isolated-r8/case_result.json`
  (`78575489e4ad34826c3f2018d49790d7163e3a87f6a9099c8a972888ac667da1`);
- transition case result:
  `run_b0_variable_prefill_repair_validation_v1/r5cot-to-r8/case_result.json`
  (`49815610b5dcf28a71fdc535d52d40498914c586de7192bbc469aaf816cd8e34`);
- isolated and transition traces:
  `472e2e06d25ec4a9dd4d82d6009048ba682ac367b4bf19592dac267803253667`
  and
  `d45e5743840216ad9b80d661ea0eaa019c5cfb8963821c55681ef49fcb1f5bf2`;
  and
- external attempt preservation:
  `p4_b0_variable_prefill_repair_validation_attempt_v1.json`
  (`448fd62d713b6b9ee68bca482558b54d11f4b51354c48e423e9521112422492f`).

Never rerun, resume, append to, or reuse this attempt or its outputs.

## Next gate

The variable-prefill runtime-evidence gate is closed at case level. The next
action is a separate source-bound V11 authorization review that binds this
immutable audit, the repaired sources, and a fresh create-only output. V11 must
not reuse V10's 72 captures or this validation output. Until that review passes,
value-screen retry, scoring, P4a engineering, action admission, and performance
claims remain unauthorized.

That separate review now passes in
`results_p4_b0_run_authorization_v11.md`. It registers the fresh
`run_b0_value_screen_v10` path and remains unexecuted. This consumed validation
output and V10's captures remain non-reusable; P4a, action admission, and
performance claims remain unauthorized.
