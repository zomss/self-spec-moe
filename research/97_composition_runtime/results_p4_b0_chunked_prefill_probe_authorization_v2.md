# Phase 97 P4 — chunked-prefill probe repair and V2 authorization

Date: 2026-08-09

Status: **path repair CPU PASS; the later GPU-4 V2 attempt was CONSUMED by a
post-boot configured/effective budget mismatch; no cohort or result exists**

## Outcome

The V1 pre-child refusal is closed without reusing its consumed authorization
or output. The probe entry point now resolves the reviewed authorization and
output paths before validation or dispatch. The normalized authorization path
is passed explicitly into parent preparation and the child command; parent and
child therefore bind the same absolute pair while persisted evidence remains
repository-relative.

The execution-path regression enters through `main()` with the exact
registered repository-relative argv and proves that dispatch receives both
reviewed paths as absolute values. The existing parent regression still
proves exactly one child launch, no scorer, the GPU-4-only environment, and
exclusive preparation creation.

## Source-bound V2 authority

`data/p4/p4_b0_chunked_prefill_probe_authorization_v2.json` binds the repaired
runner, exact-argv and parent tests, V2 validator and schema, V1 authorization
and immutable failure, live scheduler/barrier/worker sources, prompt bundle,
CPU proof, transient bound, capture runner, and consumed V9 failure.

V2 permits exactly one non-scored physical boot on GPU 4 at a fresh
create-only output:

```text
output                         run_b0_chunked_prefill_probe_v2
physical GPU                                                  4
action                                      target-matching K4
max_num_batched_tokens                                      8192
gpu_memory_utilization                                      0.90
cohorts                                      R4, R5, and R5cot
tokens per request                    one unmeasured + one measured
```

The probe passes only with positive actual CUDA-graph memory, at least 21,682
shared target-KV blocks, an atomic pure first measured decode in all three
cohorts, exact token accounting, and zero preemption, recomputation, or
invalid speculative tokens.

Retry, resume, V1-output reuse, V9-capture reuse, fallback GPU, scoring, V10,
P4a, action admission, and performance claims remain forbidden.

## Validation

```text
V2 authorization/schema/dispatch tests           17 passed
proof + capture runner + V2 authorization         58 passed, 5 subtests
K/OFF runtime                                      38 passed
live scheduler cohort                               6 passed
complete Phase 97 audit                           464 passed, 43 subtests
historical V9 run-ready assertions                  2 expected failures
Ruff check / format                                      pass
V2 source-bound validator                                pass
prepare-only exact relative CLI                           pass
V2 output at authorization time                         absent
GPU model execution                                      none
```

The two broad-suite failures are unchanged historical V9 assertions. The live
runner correctly rejects that consumed package on frozen source drift; V9 was
not modified or retried.

## Authorized command

The following command is registered but was not run in this authorization
task:

```bash
.venv/bin/python \
  research/97_composition_runtime/scripts/run_p4_b0_chunked_prefill_probe.py \
  --authorization \
  research/97_composition_runtime/data/p4/\
p4_b0_chunked_prefill_probe_authorization_v2.json \
  --output-dir \
  research/97_composition_runtime/data/p4/\
run_b0_chunked_prefill_probe_v2
```

Execution requires a separate user instruction. A passing probe would prove
only the live GPU resource and release mechanism; V10 would still require a
new source-bound decision.

## Later execution outcome

The exact command was later executed once. The engine boot and resource
precheck completed, including a 0.55 GiB actual graph pool and 24,527 shared
target-KV blocks. The first request then failed admission because the guard
compared the configured 8,192-token bound with the correctly reserve-adjusted
8,160-token effective scheduler budget. No cohort was scheduled and no result
or score was emitted.

V2 and its create-only output are consumed and immutable. See
`results_p4_b0_chunked_prefill_probe_attempt_v2.md` and
`data/p4/run_b0_chunked_prefill_probe_v2/failure.json`. The later real-config
normalization regression and guard repair pass under the separate V3 package
documented in `results_p4_b0_chunked_prefill_probe_authorization_v3.md`; V3
remains unexecuted and V10 remains unauthorized.

## Artifacts

- authorization:
  `data/p4/p4_b0_chunked_prefill_probe_authorization_v2.json`
  (`60e1f51b795a089964fa4a831879a2f7758d78f2de558f14d80b8f3ea0cb1e49`);
- validation:
  `data/p4/p4_b0_chunked_prefill_probe_authorization_v2_validation.json`
  (`7f8ea32a5705a9ef5bb79c0739e723a13399677e8a10d36f62e75a28924ec072`);
- schema:
  `schemas/p4_b0_chunked_prefill_probe_authorization_v2.schema.json`;
- validator:
  `scripts/validate_p4_b0_chunked_prefill_probe_authorization_v2.py`; and
- repaired runner and regression:
  `scripts/run_p4_b0_chunked_prefill_probe.py` and
  `tests/test_p4_b0_chunked_prefill_probe_authorization.py`.
- consumed execution record:
  `data/p4/run_b0_chunked_prefill_probe_v2/failure.json`
  (`3558f037e909f4ac418822e7f60e82e78f25b6bdd73f383549b86a7386732714`).
