# Phase 97 P4 — B0 decode-work authorization V8

Status: **AUTHORIZED and not executed. V8 binds the tested prefill-sample
offset repair to one fresh GPU-4-only, nine-boot, 432-capture value screen.
The create-only output path is absent. P4a, action admission, retry, fallback,
and performance claims remain unauthorized.**

Date: 2026-08-09.

## Repair closed by V8

V7 proved that shared target KV, request-ID canonicalization, atomic ingress,
native sampling, and the measurement resource envelope were live. Its first
cell nevertheless failed fixed-work closure because vLLM sampled one output
token during prefill. The decode-only recorder correctly excluded that event,
so `SamplingParams.max_tokens=512` yielded only 511 measured decode commits.

V8 separates frontend output work from the frozen `S_dec` currency:

```text
measured decode work       generation.max_output_tokens
unmeasured prefill sample                              1
frontend max_tokens          measured decode work + 1
recorder/scorer work          measured decode work only
```

The prompt plans, capture contract, and scorer remain unchanged. The runner
now requests one additional frontend token and validates that total output,
while the recorder still closes on the exact preregistered decode-only work.
CPU integration tests prove closure and transition through all 48 same-boot
cells for OFF, K4, and the W512 surrogate.

## Bound evidence and authority

The V8 package binds the immutable V7 authorization and incomplete-capture
failure, the passing request-ID and atomic-ingress repairs, the resource probe,
frozen prompt and scorer inputs, every execution source, GPU 4 UUID
`GPU-c9d19019-5065-2353-80a9-f1797eb19d51`, and a fresh create-only output:

```text
output path                         run_b0_value_screen_v7
physical GPU                                           4
EngineCore                                  InprocClient
prefill-sampled tokens per request                     1
capture currency                                   S_dec
physical boots / captures                         9 / 432
max_num_batched_tokens                            114688
gpu_memory_utilization                              0.96
shared target KV                                required
private draft KV                                 forbidden
```

V8 permits no GPU fallback, retry, partial resume, prior-output reuse, P4a
engineering, action admission, or production-performance claim. Any failure
must stop without scoring and requires another fresh authorization.

## Validation

The following checks passed without creating the output directory or running a
GPU model:

```text
runner + recorder focused tests          26 passed, 9 subtests
complete Phase 97 suite                404 passed, 39 subtests
vLLM K/OFF runtime tests                          38 passed
Ruff check / format                                  pass
V8 source-bound validator                            pass
GPU model execution                                  none
V8 output directory                                absent
```

The separately registered command for a future explicit launch is:

```bash
.venv/bin/python \
  research/97_composition_runtime/scripts/run_p4_b0_value_screen.py \
  --authorization \
  research/97_composition_runtime/data/p4/p4_b0_run_authorization_v8.json \
  --output-dir \
  research/97_composition_runtime/data/p4/run_b0_value_screen_v7
```

This command was not run during the V8 review.

## Artifacts

- authorization: `data/p4/p4_b0_run_authorization_v8.json`
  (`37197ba9ca87bd73dda2ada78689c40f401002ba4a55bfb7c60b8555021a6cc7`);
- validation: `data/p4/p4_b0_run_authorization_v8_validation.json`
  (`bb91f79b228c90f16dfc49d86ffc02b4e919828d45debece73b1a2a04afcc941`);
- schema: `schemas/p4_b0_run_authorization_v8.schema.json`;
- validator: `scripts/validate_p4_b0_run_authorization_v8.py`; and
- focused tests: `tests/test_p4_b0_run_authorization_v8.py`.

The next gate is a separate explicit request to execute V8 once. Validation
must pass immediately before launch, GPU 4 must be idle with the bound UUID,
and the V8 output path must remain absent.
