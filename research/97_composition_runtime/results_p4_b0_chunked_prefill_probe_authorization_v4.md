# Phase 97 P4 — query-width phase repair and V4 authorization

Date: 2026-08-09

Status: **CPU PASS; the one non-scored GPU-4 V4 authorization was later
consumed by the stopped attempt documented in
`results_p4_b0_chunked_prefill_probe_attempt_v4.md`**

## Outcome

The consumed V3 failure is repaired without reusing its authorization or
output. K4 runner evidence now distinguishes two execution phases:

- pure decode still requires draft step-0 query width exactly one;
- non-decode K4 requires an explicitly armed pure-prefill cohort, no decode
  request, and a positive draft step-0 query width; and
- both phases still require four-token draft output, an executed proposal,
  recorded draft modes, shared target KV, and shared target/draft weights.

Prefill and mixed events remain unscored. The first measured cohort event must
still be atomic pure decode and therefore retains the exact width-one check.

## Regression closure

The runtime suite now reproduces V3's 4,081-token prefill-side query width and
accepts it only for pure-prefill cohort arming. The paired negative case sends
the same width through pure decode and still fails closed against the expected
width of one. Missing and zero prefill-side widths also fail closed.

The live scheduler cohort helper now uses a wide draft query on the
pure-prefill arming event and width one on measured pure decode. OFF, K4, and
w512 retain early members and release atomically under that distinction. The
active CPU proof was revalidated against the changed runtime source; its
state-machine result and authority boundary are unchanged.

## Source-bound V4 authority

`data/p4/p4_b0_chunked_prefill_probe_authorization_v4.json` binds the phase
repair and its runtime tests, live scheduler and execution-path tests, refreshed
CPU proof, exact V3 authorization and immutable failure, probe and capture
runners, prompt bundle, resource bound, worker, and consumed V9 failure.

V4 permitted exactly one non-scored physical boot on GPU 4 using target-matching
K4, `8192 / 8160 / 32 / 0.90`, and one eight-prompt seed-0 cohort from each of
R4, R5, and R5cot. Each request performs one unmeasured prefill sample and one
measured decode token.

Retry, resume, V1/V2/V3 output reuse, V9 capture reuse, fallback GPU, scoring,
V10, P4a, action admission, and performance claims remain forbidden.

## Pre-run validation

```text
K/OFF runtime                                               42 passed
live scheduler P4 cohort                                     9 passed
complete scheduler file                                    124 passed
CPU cohort proof                                  19 passed, 2 subtests
V4 authorization/schema/dispatch                            21 passed
complete Phase 97 audit                     468 passed, 43 subtests
historical consumed-V9 run-ready assertions        2 expected failures
Ruff check / format                                               pass
V4 source-bound validator                                        pass
exact V4 prepare-only CLI                                        pass
V4 output at authorization validation                           absent
GPU execution during authorization task                           none
```

This table records the authorization-time state, before the later consumed
attempt. The two broad-audit failures are unchanged historical V9 assertions.
The live runner correctly rejects that consumed authorization on frozen source
drift.

## Executed command

The following command was later executed exactly once:

```bash
.venv/bin/python \
  research/97_composition_runtime/scripts/run_p4_b0_chunked_prefill_probe.py \
  --authorization \
  research/97_composition_runtime/data/p4/\
p4_b0_chunked_prefill_probe_authorization_v4.json \
  --output-dir \
  research/97_composition_runtime/data/p4/\
run_b0_chunked_prefill_probe_v4
```

The child completed all three cohort barriers, then failed its probe-only final
request-ID comparison because valid randomized internal IDs were not
canonicalized. V4 is consumed without retry, resume, fallback, scoring, or a
probe result. V10 remains unauthorized.

## Artifacts

- authorization:
  `data/p4/p4_b0_chunked_prefill_probe_authorization_v4.json`
  (`da86335dc02c6b96c873c6b8ee800be69826bb3450b28f113f502ea3d34a3499`);
- validation:
  `data/p4/p4_b0_chunked_prefill_probe_authorization_v4_validation.json`
  (`5a9a058957f5c235b84b18ea52e665ef41e33ec9179c4f00f73faa9b1c53497e`);
- consumed-attempt report:
  `results_p4_b0_chunked_prefill_probe_attempt_v4.md`;
- immutable failure:
  `data/p4/run_b0_chunked_prefill_probe_v4/failure.json`
  (`a12bb41ddc31ceb96f35fad9e750d95908045bc6d4b7ff41634c2202894fde0e`);
- schema:
  `schemas/p4_b0_chunked_prefill_probe_authorization_v4.schema.json`;
- validator:
  `scripts/validate_p4_b0_chunked_prefill_probe_authorization_v4.py`; and
- repaired execution path:
  `vllm/v1/spec_decode/koff_runtime.py`,
  `tests/v1/spec_decode/test_koff_runtime.py`,
  `tests/v1/core/test_scheduler.py`, and
  `scripts/run_p4_b0_chunked_prefill_probe.py`.

The request-ID repair and separate unexecuted V5 authority are documented in
`results_p4_b0_chunked_prefill_probe_authorization_v5.md`.
