# Phase 97 P4 — B0 request-ID authorization V7

Status: **AUTHORIZED and not executed. V7 binds the tested request-ID
canonicalization repair to one fresh GPU-4-only, nine-boot, 432-capture value
screen. The create-only output path is absent. P4a, action admission, retry,
fallback, and performance claims remain unauthorized.**

Date: 2026-08-09.

## Repair closed by V7

V6 queued all eight requests atomically, but vLLM converted each frozen
external prompt ID into an internal ID ending in eight lowercase hexadecimal
characters. The atomic-ingress proof accepted any suffix, while the recorder
accepted only exact frozen IDs. The first live event therefore failed closed
before capture, adaptation, or scoring.

V7 preserves vLLM's internal uniqueness mechanism. One shared helper now:

- accepts an exact frozen request ID;
- otherwise accepts only `<frozen-id>-<exactly 8 lowercase hex>`;
- maps accepted internal IDs back to their frozen external IDs in row order;
- rejects malformed suffixes, unknown IDs, reordered proof rows, and two
  internal IDs that canonicalize to one frozen ID; and
- writes only frozen IDs into scoreable captures.

The same helper is used by the live same-event recorder and the atomic-ingress
trace analyzer. `VLLM_DISABLE_REQUEST_ID_RANDOMIZATION` is not enabled.

## Bound evidence and authority

The V7 package binds the immutable V6 authorization and zero-capture failure,
the passing non-scored atomic-ingress proof, the resource repair, frozen prompt
and scorer inputs, the input-processor suffix source, both canonicalization
consumers and their tests, the live runner, GPU 4 UUID
`GPU-c9d19019-5065-2353-80a9-f1797eb19d51`, and a fresh create-only output:

```text
output path                         run_b0_value_screen_v6
physical GPU                                           4
EngineCore                                  InprocClient
request-ID randomization                        retained
capture request IDs                       frozen external
physical boots / captures                         9 / 432
max_num_batched_tokens                            114688
gpu_memory_utilization                              0.96
shared target KV                                required
private draft KV                                 forbidden
```

V7 permits no GPU fallback, retry, partial resume, reuse of V6 output, P4a
engineering, action admission, or production-performance claim. Any failure
must stop without scoring and requires another fresh authorization.

## Validation

The following checks passed without creating the output directory or running a
GPU model:

```text
request-ID recorder/proof focused tests          19 passed
V7 authorization focused tests                    9 passed
complete Phase 97 suite                         392 passed
vLLM K/OFF runtime tests                         38 passed
Ruff check / format                                  pass
GPU model execution                                  none
V7 output directory                                absent
```

The separately registered command for a future explicit launch is:

```bash
.venv/bin/python \
  research/97_composition_runtime/scripts/run_p4_b0_value_screen.py \
  --authorization \
  research/97_composition_runtime/data/p4/p4_b0_run_authorization_v7.json \
  --output-dir \
  research/97_composition_runtime/data/p4/run_b0_value_screen_v6
```

This command was not run during the V7 review.

## Artifacts

- authorization: `data/p4/p4_b0_run_authorization_v7.json`
  (`fabb3fad1fb689758ccb899c3be440e4ea47c4821564a10a1090dbbae67981df`);
- validation: `data/p4/p4_b0_run_authorization_v7_validation.json`
  (`595caa04ad12d5d475a442d2a22470accd9c9ab22a94ef0e9bdcc406cd3a9895`);
- schema: `schemas/p4_b0_run_authorization_v7.schema.json`;
- validator: `scripts/validate_p4_b0_run_authorization_v7.py`; and
- focused tests: `tests/test_p4_b0_run_authorization_v7.py`.

The next gate is a separate explicit request to execute V7 once. Validation
must still pass immediately before launch, GPU 4 must be idle with the bound
UUID, and the V7 output path must remain absent.
