# Phase 97 P4 — B0 value-screen authorization V6

Status: **AUTHORIZED and not executed. V6 binds the scored runner to the
passing in-process atomic-ingress mechanism for one GPU-4-only, nine-boot,
432-capture value screen. The output directory is absent. P4a, action
admission, and performance claims remain unauthorized.**

Date: 2026-08-09.

## Repair closed by V6

V5 failed with zero complete captures because multiprocess EngineCore
admission allowed one request to decode while seven requests were still in
prefill. The separately consumed V2 atomic proof established that
`InprocClient` queues all eight requests before execution and produces the
required two-step geometry.

V6 wires that mechanism into the scored matrix runner:

- every boot forces `VLLM_ENABLE_V1_MULTIPROCESSING=0` before importing vLLM;
- the boot specification cannot override the setting;
- a subprocess preflight proves the environment selects in-process mode;
- the child asserts that the constructed EngineCore is `InprocClient`;
- each capture microbatch checks that every request is queued before the first
  `engine.step()`; and
- native-sampler, in-process EngineCore, and physical-GPU checks all run before
  the create-only output directory is made.

The queueing logic remains otherwise unchanged. All requests in a chunk are
added before the existing completion loop begins.

## Bound evidence

The V6 package directly binds:

- the consumed V5 authorization and immutable zero-capture failure;
- the passing, non-scored atomic-ingress authorization, proof, and trace;
- the full-prefill resource repair (`22,113` probe blocks and a `21,682`
  launch floor);
- the live proof capacity (`22,090` shared target-KV blocks);
- the exact prompt manifest, compressed token bundle, and scorer contract;
- the runner, adapter, scorer, runtime, scheduler, EngineCore, shared-KV,
  draft, sampler, validator, schema, and test source hashes; and
- physical GPU 4 UUID
  `GPU-c9d19019-5065-2353-80a9-f1797eb19d51`, with no fallback.

The effective measurement contract remains:

```text
EngineCore                                      InprocClient
VLLM_ENABLE_V1_MULTIPROCESSING                           0
max_num_batched_tokens                             114,688
effective scheduler budget                        114,656
gpu_memory_utilization                               0.96
chunked prefill                                       true
physical boots / captures                          9 / 432
shared target KV required                              true
private draft KV allowed                              false
```

The real-serving boundary remains `8192 / 0.90` with natural mixed prefill and
decode. V6 does not promote the measurement envelope or in-process harness to
a serving default.

## Authority boundary

V6 authorizes exactly one execution at the registered path:

```bash
.venv/bin/python \
  research/97_composition_runtime/scripts/run_p4_b0_value_screen.py \
  --authorization \
  research/97_composition_runtime/data/p4/p4_b0_run_authorization_v6.json \
  --output-dir \
  research/97_composition_runtime/data/p4/run_b0_value_screen_v5
```

This command has **not** been run. Before execution, validation must be rerun,
the source hashes and absent output must still match, and GPU 4 must have the
registered identity with no active compute process. Any failure consumes V6;
retry and partial resume are forbidden.

A complete run requires all 432 nonempty captures before adaptation or
scoring. The score cannot grant authority. Even a value pass can only support
a later, separately reviewed P4a package.

## Validation

The V6 validator passed without creating the output or executing a GPU model:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research/97_composition_runtime/scripts/validate_p4_b0_run_authorization_v6.py \
  --authorization \
  research/97_composition_runtime/data/p4/p4_b0_run_authorization_v6.json
```

Validation results:

```text
capture-runner focused tests                         19 passed
V6 authorization focused tests                      15 passed
complete Phase 97 suite                            376 passed
Phase 97 subtests                                    21 passed
Ruff check / format                                      pass
GPU model execution                                      none
V6 output directory                                    absent
```

## Artifacts

- authorization: `data/p4/p4_b0_run_authorization_v6.json`
  (`f3348821c2f8c8ed6bbb74df620c51a33e88eec3b3b0971cc64637e664c4afca`);
- validation: `data/p4/p4_b0_run_authorization_v6_validation.json`
  (`b47dca8f5cbc64e981750139e141bd0c3ba14a824c7819c52cce6ab0b5374d81`);
- schema: `schemas/p4_b0_run_authorization_v6.schema.json`;
- validator: `scripts/validate_p4_b0_run_authorization_v6.py`; and
- focused tests: `tests/test_p4_b0_run_authorization_v6.py`.

The next gate is the separately requested one-shot V6 execution. This review
does not itself consume V6.
