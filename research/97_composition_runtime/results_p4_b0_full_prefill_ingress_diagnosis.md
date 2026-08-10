# Phase 97 P4 — full-prefill ingress diagnosis

Status: **PASS for diagnosis only. The consumed V5 first event was excluded
only because it mixed one decode with seven prefills. There was no preemption,
recomputation, or invalid-spec-token exclusion. The value screen remains
failed and unscored, and V6 is not authorized.**

Date: 2026-08-09.

## Contract

The source-bound package authorized one non-scored boot on GPU 4. It reused
the exact consumed V5 OFF boot specification, first capture plan, model,
prompt records, and measurement-only resource envelope:

```text
max_num_batched_tokens                            114,688
effective scheduler budget                       114,656
gpu_memory_utilization                              0.96
chunked prefill                                      true
prompt workload                     first frozen R4/seed-0 batch
prompt count / tokens                         8 / 65,200
P4 recorder                                      enabled
passive K/OFF trace                              enabled
adaptation / scoring                       forbidden / forbidden
```

The passive trace was flushed immediately before the strict P4 recorder. The
expected successful diagnostic disposition was therefore a preserved trace
followed by the recorder rejecting the same score-ineligible event.

## Live result

The run produced exactly two target-step records:

```text
step 0   request p000 pure prefill                q=[8077]
step 1   p000 decode plus seven prefills          q=[1, 7839, 8368,
                                                     8435, 7836, 7876,
                                                     8597, 8172]
step 1 exclusion                                  prefill_or_mixed_batch
preemptions / recomputed tokens                   0 / 0
draft dispatched                                  false
shared target-KV capacity                         22,090 blocks
```

The second record is the exact V5 ingress failure geometry. The first request
had completed all 8,077 prompt tokens and entered width-one decode, while the
other seven requests still had zero computed tokens and scheduled their full
57,123 prompt tokens. Its exclusion list contained only
`prefill_or_mixed_batch`.

The recorder then failed closed as expected:

```text
complete captures                                 0
zero-byte placeholders                            1
adapted rounds                                    0
score emitted                                 false
```

All 36 draft attention layers continued to alias the target-owned KV tensors,
no private draft KV was allocated, and all 291 draft parameters continued to
alias target parameters.

## Cause

The full-prefill budget controls how much already-admitted work can be
scheduled; it does not atomically admit the intended microbatch. In the
multiprocess `LLMEngine` path, each `add_request` is sent immediately to the
EngineCore, whose busy loop may begin once the first request arrives. The
trace shows exactly that ordering: p000 was scheduled alone before p001-p007
reached the same scheduling boundary.

Therefore V5's premise was wrong even though 65,200 tokens fit comfortably
inside 114,656. Increasing the token budget cannot by itself guarantee a
pure-prefill first step or a pure-decode first score-bearing step.

## Decision and next gate

The diagnosis is complete, but the screen is not repaired. Do not change the
recorder to ignore mixed events: those events already emit target output and
dropping them would violate requested-output and same-event counter closure.

Before any V6 authorization, add a measurement-harness ingress barrier that
makes the eight requests visible to the scheduler before execution begins.
The narrow candidate is the existing in-process EngineCore mode
(`VLLM_ENABLE_V1_MULTIPROCESSING=0`), where `add_request` queues work and
`engine.step()` starts execution. First cover that contract in CPU tests, then
use a separately authorized one-boot, non-scored GPU-4 proof. That proof must
observe all eight prompts in the initial pure-prefill step and all eight
width-one rows in the first pure-decode step, with no P4 exclusion and with
the same shared-KV and weight-alias invariants. Only then may a fresh screen
package be reviewed.

The real serving boundary remains `8192 / 0.90` with natural mixed prefill and
decode. The in-process barrier is a measurement-harness proposal, not a
serving default.

## Validation

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  research/97_composition_runtime/tests/test_p4_b0_full_prefill_ingress_diagnosis.py \
  -q

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  research/97_composition_runtime/tests -q
```

The focused suite passes 9 tests. The complete Phase 97 suite passes 346 tests
plus 21 subtests. Ruff check and format pass the new runner and focused test.
GPU 4 returned to 0 MiB with no compute process.

## Artifacts

- authorization:
  `data/p4/p4_b0_full_prefill_ingress_diagnosis_authorization.json`;
- result:
  `data/p4/run_b0_full_prefill_ingress_diagnosis_v1/diagnosis.json`;
- raw trace:
  `data/p4/run_b0_full_prefill_ingress_diagnosis_v1/koff_trace.jsonl`;
- fail-closed placeholder:
  `data/p4/run_b0_full_prefill_ingress_diagnosis_v1/captures/capture-b1-p1-off-r4-s0-r1.json`;
  and
- preparation evidence:
  `data/p4/run_b0_full_prefill_ingress_diagnosis_v1/preparation.json`.

The authorization is consumed and the output is immutable. It must not be
rerun, resumed, or used as score evidence.
