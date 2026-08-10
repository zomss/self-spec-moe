# Phase 97 P4 — B0 full-prefill resource-repair probe

Status: **PASS for the measurement-only capacity gate. At
`gpu_memory_utilization=0.96`, GPU 4 exposes 22,113 shared-KV blocks, 431
above the 21,682-block launch floor. The probe executed zero requests and is
consumed. It does not validate or promote the real-serving configuration.**

Date: 2026-08-09.

## Result

The exact one-initialization probe changed only GPU memory utilization from
the prior failed probe. It retained the full-prefill measurement budget,
chunked prefill, native sampler, target-owned shared KV, and frozen model and
prompt contracts.

```text
physical GPU                                      4
max_num_batched_tokens                      114,688
chunked prefill                                  true
gpu_memory_utilization                           0.96
effective scheduler token budget              114,656
maximum frozen microbatch prompt tokens        112,908
shared-KV capacity tokens                      353,808
shared-KV blocks                                22,113
conservative launch floor                      21,682
headroom above launch floor                       431
requests executed                                   0
generation performed                             false
```

The prior `0.90` probe exposed 19,928 blocks. Raising the measurement-only
utilization to `0.96` recovered 2,185 blocks, exceeded the 22,102-block
projection by 11, and closed the registered floor.

## Boundary preserved

This result is capacity evidence for the frozen value-screen shape, not a
serving recommendation. The real-serving diagnosis remains separately fixed
at:

```text
max_num_batched_tokens                         8,192
chunked prefill                                  true
gpu_memory_utilization                           0.90
mixed prefill/decode diagnosis required          true
```

The `114,688 / 0.96` settings must not become the serving default. A later
real mixed prefill/decode diagnosis must use `8192 / 0.90` after the value
screen.

## Fail-closed disposition

The machine-readable result is
`data/p4/p4_b0_full_prefill_resource_repair_probe.json`. The probe itself
grants no value-screen, P4a, action-admission, or performance authority. Its
create-only authorization is consumed and cannot be reused. The passing result
may support a separate source-bound V5 review.

## Validation

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  research/97_composition_runtime/tests/test_p4_b0_full_prefill_resource_repair_probe.py \
  -q

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  research/97_composition_runtime/tests -q
```

The complete Phase 97 suite passes 328 tests plus 21 subtests. The Phase 97
Ruff check and format hooks pass.

## Next step

Use this result only as evidence in the separate V5 value-screen
authorization. Preserve the `8192 / 0.90` real-serving boundary for the later
mixed prefill/decode diagnosis.
