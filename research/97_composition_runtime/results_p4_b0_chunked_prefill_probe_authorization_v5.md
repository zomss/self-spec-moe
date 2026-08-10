# Phase 97 P4 — probe request-ID repair and V5 authorization

Date: 2026-08-09

Status: **CPU PASS; the one source-bound, non-scored GPU-4 V5 probe was later
executed exactly once and PASSED; V5 is consumed and V10 remains
unauthorized**

## Outcome

The consumed V4 attempt completed all three GPU cohort barriers but failed its
probe-only final identity comparison. The live engine correctly retained
randomized internal request IDs such as `R4-s0-p000-a3388847`; the lightweight
probe recorder retained those IDs, while its final validator compared them
directly with frozen manifest IDs.

The repaired `_validate_probe_evidence` now calls the existing
`canonicalize_p4_request_ids` helper. It preserves randomized runtime IDs,
validates their exact eight-lowercase-hex suffix contract, rejects foreign or
colliding identities, and returns a copied event with canonical frozen IDs for
the result artifact. The original scheduler event is not mutated.

Direct regressions prove valid suffixed IDs in non-manifest scheduler order and
fail closed on malformed suffixes, foreign identities, and post-normalization
collisions. V4's phase-aware wide-prefill and width-one pure-decode behavior is
unchanged.

## Source-bound V5 authority

`data/p4/p4_b0_chunked_prefill_probe_authorization_v5.json` binds the exact V4
authorization and immutable failure, request-ID repair and tests, current probe
runner, scheduler and K/OFF runtime, active cohort proof, prompt bundle,
resource bound, worker, capture runner, and consumed V9 failure.

V5 permitted exactly one non-scored physical boot on GPU 4 using target-matching
K4, `8192 / 8160 / 32 / 0.90`, and one eight-prompt seed-0 cohort from each of
R4, R5, and R5cot. It used a new create-only
`run_b0_chunked_prefill_probe_v5` output.

Retry, resume, V1/V2/V3/V4 output reuse, V9 capture reuse, fallback GPU,
scoring, V10, P4a, action admission, and performance claims remain forbidden.

## Pre-run validation

```text
probe request-ID positive/fail-closed regressions             4 passed
V5 authorization/schema/dispatch                             27 passed
K/OFF runtime                                                 42 passed
live scheduler P4 cohort                                       9 passed
complete scheduler file                                      124 passed
CPU cohort proof                                    19 passed, 2 subtests
complete Phase 97 audit                       474 passed, 43 subtests
historical consumed-V9 run-ready assertions          2 expected failures
Ruff check / format                                                 pass
V5 source-bound validator                                          pass
exact V5 prepare-only CLI                                          pass
V5 output                                                        absent
GPU execution in this task                                         none
```

The two broad-audit failures are unchanged historical V9 assertions. The live
runner correctly rejects that already-consumed authorization on frozen source
drift. No active V5 test failed.

## Executed command

The following command was later executed exactly once:

```bash
.venv/bin/python \
  research/97_composition_runtime/scripts/run_p4_b0_chunked_prefill_probe.py \
  --authorization \
  research/97_composition_runtime/data/p4/\
p4_b0_chunked_prefill_probe_authorization_v5.json \
  --output-dir \
  research/97_composition_runtime/data/p4/\
run_b0_chunked_prefill_probe_v5
```

The child completed all three R4/R5/R5cot cohort barriers and emitted the
formal non-scored `probe_result.json`. All 24 result rows use canonical frozen
request IDs, all three measured events have zero quality violations, and the
24,527-block shared target-KV capacity remains above the 21,682-block floor.
See `results_p4_b0_chunked_prefill_probe_attempt_v5.md` for the immutable run
record. V10 still requires a separate source-bound decision.

## Artifacts

- authorization:
  `data/p4/p4_b0_chunked_prefill_probe_authorization_v5.json`
  (`795ead82642faaed18f59e759f870aed86dd4c769d5bd4d80344b4003c1d8599`);
- validation:
  `data/p4/p4_b0_chunked_prefill_probe_authorization_v5_validation.json`
  (`446550579488566d1d63239e882715fd21115d9b5e627d0986d84399aad4f982`);
- consumed-attempt report:
  `results_p4_b0_chunked_prefill_probe_attempt_v5.md`;
- immutable probe result:
  `data/p4/run_b0_chunked_prefill_probe_v5/probe_result.json`
  (`dc398883c6ed62a81ee46ff5643a68216954983d5b7787951218463149120cc2`);
- schema:
  `schemas/p4_b0_chunked_prefill_probe_authorization_v5.schema.json`;
- validator:
  `scripts/validate_p4_b0_chunked_prefill_probe_authorization_v5.py`;
- repaired probe runner:
  `scripts/run_p4_b0_chunked_prefill_probe.py`; and
- direct regressions:
  `tests/test_p4_b0_chunked_prefill_probe_authorization.py`.
