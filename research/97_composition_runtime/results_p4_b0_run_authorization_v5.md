# Phase 97 P4 — B0 full-prefill resource retry authorization V5

Status: **APPROVE for one fresh GPU-4 OFF/K4/w512 value-screen attempt. The
measurement-only engine uses `114688 / 0.96` with chunked prefill; the real
serving `8192 / 0.90` chunked-prefill diagnosis remains separate and pending.
P4a, action admission, and performance claims remain unauthorized.**

Date: 2026-08-09.

## Decision

V4 stopped in its first OFF cell with zero complete captures because the
8,160-token effective scheduler budget admitted a mixed prefill/decode event,
which the strict pure-decode recorder correctly rejected. The full-prefill
measurement budget raises the effective budget to 114,656 tokens, covering
the frozen 112,908-token maximum microbatch and speculative reserve.

The first full-prefill capacity probe at `gpu_memory_utilization=0.90` failed
with 19,928 shared-KV blocks. The separately authorized repair probe changed
only that measurement field to `0.96` and passed with 22,113 blocks, 431 above
the 21,682-block launch floor. It initialized one engine and executed zero
requests.

V5 binds that evidence and the current runner, tests, conformance validator,
sampler backend, repair-probe code, schema, validator, and tests. It preserves
all previous failed outputs without resume or overwrite.

```text
physical GPU                                      4 only
physical boots                                    9
cells per boot                                   48
required complete captures                      432
measurement max_num_batched_tokens          114,688
measurement gpu_memory_utilization              0.96
measurement chunked prefill                     true
shared-KV launch floor                         21,682 blocks
measured shared-KV capacity                    22,113 blocks
fresh output                 run_b0_value_screen_v4
fallback GPU, reuse, or partial resume       forbidden
serving diagnosis authority                    false
P4a or admission authority                     false
```

## Exact authorized invocation

```bash
.venv/bin/python \
  research/97_composition_runtime/scripts/run_p4_b0_value_screen.py \
  --authorization \
  research/97_composition_runtime/data/p4/p4_b0_run_authorization_v5.json \
  --output-dir \
  research/97_composition_runtime/data/p4/run_b0_value_screen_v4
```

No other GPU, output path, engine override, retry, or serving diagnosis is
authorized. Any failure stops without scoring and requires a fresh
authorization.

## Real-serving boundary

V5 explicitly records, but does not authorize, the later real-serving
diagnosis:

```text
max_num_batched_tokens                         8,192
chunked prefill                                  true
gpu_memory_utilization                           0.90
mixed prefill/decode diagnosis required          true
```

The measurement-only `114688 / 0.96` repair is not a serving default. The
`8192 / 0.90` diagnosis remains required after this screen.

## Validation

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research/97_composition_runtime/scripts/validate_p4_b0_run_authorization_v5.py \
  --authorization \
  research/97_composition_runtime/data/p4/p4_b0_run_authorization_v5.json

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  research/97_composition_runtime/tests/test_p4_b0_run_authorization_v5.py -q

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  research/97_composition_runtime/tests -q
```

The V5 suite passes 17 tests. The complete Phase 97 suite passes 328 tests
plus 21 subtests. The Phase 97 Ruff check and format hooks pass. Validation
reports current source bindings, nine boots, 432 captures, native sampling,
22,113 measured blocks, and an absent fresh output directory. No model or
screen was executed while preparing V5.

## Next step

Run only the exact V5 invocation. A successful run must emit all 432 captures,
adapted rounds, and a score before producing `p4_b0_value_screen_result`.
Afterward, diagnose real mixed prefill/decode behavior separately at the
preserved `8192 / 0.90` chunked-prefill configuration. Neither result by
itself authorizes P4a or admits w512.
