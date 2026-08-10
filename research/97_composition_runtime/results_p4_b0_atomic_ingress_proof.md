# Phase 97 P4 — B0 atomic-ingress proof

Status: **PASS for the measurement-only atomic-ingress gate. The in-process
EngineCore queued all eight requests before execution and produced the exact
two-step prefill/decode geometry. This result permits a V6 authorization
review only; V6 execution, scoring, P4a, and action admission remain
unauthorized.**

Date: 2026-08-09.

## Contract

The source-bound V2 package authorized one non-scored boot on physical GPU 4.
It reused the exact first V5 R4/seed-0 OFF workload and measurement envelope,
but selected the in-process EngineCore and disabled the P4 capture, adapter,
and scorer:

```text
EngineCore                                      InprocClient
VLLM_ENABLE_V1_MULTIPROCESSING                           0
max_num_batched_tokens                             114,688
effective scheduler budget                        114,656
gpu_memory_utilization                               0.96
chunked prefill                                       true
prompt count / tokens                          8 / 65,200
engine steps                                             2
P4 capture / adaptation / scoring       disabled / forbidden / forbidden
passive K/OFF trace                         enabled, non-scored
```

All eight `add_request` calls had to complete before the first `engine.step()`.
The proof required one eight-request pure-prefill step followed by one
eight-request width-one pure-decode OFF step with no exclusion.

## Attempt history

The V1 authorization was consumed without loading model weights or executing
an engine step. Its child environment assigned an empty string to the integer
`VLLM_SELF_SPEC_P4_MIN_KV_BLOCKS` setting, and environment parsing raised
`ValueError`. V1 emitted no trace, capture, or score and was not retried.

The repair changed only that disabled capture-conformance setting from `""`
to `"0"`, added a regression test against the real environment parser, and
required a fresh V2 authorization and output path.

## V2 live result

V2 passed the complete proof contract:

```text
step 0 requests / query widths       8 / [8077, 7839, 8368, 8435,
                                           7836, 7876, 8597, 8172]
step 1 requests / query widths       8 / [1, 1, 1, 1, 1, 1, 1, 1]
step 1 action / exclusions                           OFF / []
preemptions / recomputed tokens                       0 / 0
draft dispatched                                      false
target/draft weight aliases                             291
shared target-KV layers                                  36
private draft KV allocated                            false
shared target-KV capacity                     22,090 blocks
```

The initial pure-prefill record carried the expected `no_decode_action` and
`prefill_or_mixed_batch` labels. The next record was a pure width-one decode
for all eight requests, was replay-eligible, committed all eight target steps,
and had no exclusion. The target-owned KV identity remained unchanged between
the two records.

## Decision and boundary

The earlier diagnosis is confirmed: the V5 mixed ingress came from
multiprocess request admission, not insufficient scheduler-token budget.
In-process EngineCore mode supplies the required measurement-harness barrier
for this exact workload because requests are queued locally and execution
starts only when `engine.step()` is called.

This is not a serving recommendation and does not repair the scored screen by
itself. The machine result intentionally leaves these claims false:

- `value_screen_repaired`;
- `v6_authorized`;
- `performance_claim_allowed`; and
- `action_admitted`.

The next gate is a fresh, source-bound V6 authorization review. It must bind
the value-screen runner to in-process EngineCore mode for every boot, assert
that mode before request admission, preserve the queue-all-before-step path,
and bind fresh runner/test hashes and a create-only output. Passing this proof
does not itself authorize that run.

The real serving boundary remains the separately diagnosed `8192 / 0.90`
mixed-prefill configuration. The `114688 / 0.96` envelope and in-process
barrier remain measurement-only.

## Validation

Before execution, the focused suite passed 8 tests with 2 expected
post-artifact skips, and authorization validation passed without creating the
output. After execution:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  research/97_composition_runtime/tests/test_p4_b0_atomic_ingress_proof.py \
  -q

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  research/97_composition_runtime/tests -q

.venv/bin/pre-commit run ruff-check --files \
  research/97_composition_runtime/scripts/run_p4_b0_atomic_ingress_proof.py \
  research/97_composition_runtime/tests/test_p4_b0_atomic_ingress_proof.py

.venv/bin/pre-commit run ruff-format --files \
  research/97_composition_runtime/scripts/run_p4_b0_atomic_ingress_proof.py \
  research/97_composition_runtime/tests/test_p4_b0_atomic_ingress_proof.py
```

The focused suite passed 10 tests. The complete Phase 97 suite passed 356
tests plus 21 subtests. Both Ruff hooks passed. GPU 4 returned to 0 MiB with
the authorized UUID and no fallback execution.

## Artifacts

- V2 authorization:
  `data/p4/p4_b0_atomic_ingress_proof_authorization_v2.json`
  (`9811fc0c2a22edd7a127df4a405bd9d66263106becbc000321e54c4b26155983`);
- V2 preparation:
  `data/p4/run_b0_atomic_ingress_proof_v2/preparation.json`
  (`7be79049c685aea3a79865504232f550ec8fc9ec36ed0868fa095add4b8438ca`);
- V2 proof:
  `data/p4/run_b0_atomic_ingress_proof_v2/proof.json`
  (`d3c92a20d9fa41ed92131b1b704b32c4f2a64b984574cfe37b18d545292307e2`);
- V2 passive trace:
  `data/p4/run_b0_atomic_ingress_proof_v2/koff_trace.jsonl`
  (`09f7e070987827a2970e2e9bc413c4083267ed65231b6acae9ab820d02bec8bc`);
  and
- V1 failure diagnosis:
  `data/p4/p4_b0_atomic_ingress_proof_v1_failure_diagnosis.json`.

Both authorizations are consumed and both output directories are immutable.
Neither attempt may be rerun, resumed, or used as scored evidence.
