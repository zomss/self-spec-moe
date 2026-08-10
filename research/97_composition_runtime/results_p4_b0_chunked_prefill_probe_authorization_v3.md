# Phase 97 P4 — chunked-prefill budget repair and V3 authorization

Date: 2026-08-09

Status: **CPU PASS; the later one-shot GPU-4 V3 probe was CONSUMED by a
prefill/decode query-width phase mismatch; no complete cohort, probe result,
or score exists**

## Outcome

The V2 refusal is repaired without reusing its consumed authorization or
output. Cohort admission now distinguishes the configured compiled bound from
the effective scheduler budget:

```text
configured max_num_batched_tokens       8192
draft-model speculative slot reserve      32
effective max_num_scheduled_tokens       8160
configured max_num_seqs                    32
```

The admission gate requires all four values and synchronous chunked prefill.
The K/OFF trace, cohort history, probe resource evidence, and probe-result
validator now report configured and effective token budgets under separate
field names.

## Regression closure

The live cohort tests now instantiate a real target-matching draft model. This
executes `VllmConfig._set_max_num_scheduled_tokens` and observes 8,160 instead
of the previous n-gram helper's 8,192 fallback. Positive OFF, K4, and w512
cases retain early members and release atomically. Negative cases independently
reject configured-budget, effective-budget, and sequence-capacity drift.

The complete scheduler suite exposed one partial-construction unit test that
did not initialize the phase-local cohort attributes. The output hook now uses
a no-op-safe lookup when no cohort state exists; ordinary serving behavior is
unchanged.

## Source-bound V3 authority

`data/p4/p4_b0_chunked_prefill_probe_authorization_v3.json` binds the repaired
scheduler, real-normalization test helper and tests, probe runner, V3 schema
and validator, immutable V2 failure, V2 authorization, prompt bundle, live
barrier and worker sources, CPU proof, transient bound, capture runner, and
consumed V9 failure.

V3 permits exactly one non-scored physical boot on GPU 4 using target-matching
K4, `8192 / 8160 / 32 / 0.90`, and one eight-prompt seed-0 cohort from each of
R4, R5, and R5cot. Each request performs one unmeasured prefill sample and one
measured decode token.

Retry, resume, V1/V2 output reuse, V9 capture reuse, fallback GPU, scoring,
V10, P4a, action admission, and performance claims remain forbidden.

## Validation

```text
live scheduler P4 cohort tests                         9 passed
complete scheduler file                             124 passed
V3 authorization/schema/dispatch tests               19 passed
proof + capture runner + V3 authorization             60 passed, 5 subtests
K/OFF runtime                                         18 passed, 8 subtests
complete Phase 97 audit                              466 passed, 43 subtests
historical consumed-V9 run-ready assertions            2 expected failures
Ruff check / format                                         pass
V3 source-bound validator                                  pass
exact V3 prepare-only CLI                                  pass
V3 output                                                absent
GPU execution in this task                                 none
```

The two broad-audit failures are unchanged historical V9 assertions. The live
runner correctly rejects that consumed package on frozen source drift.

## Authorized command

The following command is registered but was not executed in this task:

```bash
.venv/bin/python \
  research/97_composition_runtime/scripts/run_p4_b0_chunked_prefill_probe.py \
  --authorization \
  research/97_composition_runtime/data/p4/\
p4_b0_chunked_prefill_probe_authorization_v3.json \
  --output-dir \
  research/97_composition_runtime/data/p4/\
run_b0_chunked_prefill_probe_v3
```

A passing probe would still require a separate V10 decision. Any failure must
stop without retry, resume, fallback, or scoring.

## Later execution outcome

The exact command was later executed once. Engine initialization, the repaired
`8192 / 8160 / 32` budget gate, the 0.55 GiB actual graph pool, and the
24,527-block shared target-KV capacity all passed. On the first R4
chunked-prefill event, scheduler metadata correctly reported no decode rows,
`pure_decode=false`, and K4 arming for the next action. The draft proposer
reported its prefill-side step-0 query width as 4,081.

The runner evidence builder incorrectly enforced K4's width-one decode
invariant on that non-decode event and failed closed. V3 and its create-only
output are consumed and immutable; no retry, resume, fallback, result, or score
was attempted. See `results_p4_b0_chunked_prefill_probe_attempt_v3.md` and
`data/p4/run_b0_chunked_prefill_probe_v3/failure.json`. Any repair requires
CPU regressions, a fresh source-bound V4 authorization, and a new output path.
That repair and CPU review later passed under the separate V4 package in
`results_p4_b0_chunked_prefill_probe_authorization_v4.md`. V4 remains
unexecuted; V10 remains unauthorized.

## Artifacts

- authorization:
  `data/p4/p4_b0_chunked_prefill_probe_authorization_v3.json`
  (`9e15f80a9eb4abfbc73e24cb72f7cdd7ea1bdd96c79824b2be0aeca3dc27f3f6`);
- validation:
  `data/p4/p4_b0_chunked_prefill_probe_authorization_v3_validation.json`
  (`136274a130a2106437cd80e7ed0a42ba919bf7931c49bb0fa6fdb0e975522f1b`);
- schema:
  `schemas/p4_b0_chunked_prefill_probe_authorization_v3.schema.json`;
- validator:
  `scripts/validate_p4_b0_chunked_prefill_probe_authorization_v3.py`; and
- repaired execution path:
  `vllm/v1/core/sched/scheduler.py`, `tests/v1/core/utils.py`,
  `tests/v1/core/test_scheduler.py`, and
  `scripts/run_p4_b0_chunked_prefill_probe.py`; and
- consumed execution record:
  `data/p4/run_b0_chunked_prefill_probe_v3/failure.json`
  (`7117131be7c0508c1fd454785e1c67a708b1b4a582f6bd0ae6f124bf79c655ab`).
