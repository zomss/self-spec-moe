# Phase 97 P4 — B0 full-prefill resource probe

Status: **FAIL. The measurement-only full-prefill engine initializes on GPU 4,
but its 19,928 shared-KV blocks are below both the 21,682-block launch floor
and the 21,000-block live-workload requirement. The one-initialization probe
authorization is consumed. V5 and the 432-cell value screen remain blocked.**

Date: 2026-08-09.

## Result

The exact authorized probe initialized one engine and executed zero requests.
It retained chunked prefill, the PyTorch-native sampler, 291 target/draft
weight aliases, and all 36 draft attention layers on target-owned KV. No
private draft KV was allocated.

```text
physical GPU                                      4
max_num_batched_tokens                      114,688
effective scheduler token budget            114,656
maximum frozen microbatch prompt tokens      112,908
full-prefill token headroom                     1,748
KV block size                                      16
measured shared-KV capacity                  318,848 tokens
measured shared-KV blocks                     19,928
frozen live-workload requirement              21,000 blocks
conservative launch floor                     21,682 blocks
shortfall versus live requirement              1,072 blocks
shortfall versus launch floor                  1,754 blocks
requests executed                                  0
generation performed                            false
```

The machine-readable result is
`data/p4/p4_b0_full_prefill_resource_probe.json`.

## Interpretation

Chunked prefill remains enabled. The larger scheduler budget is required only
for this strict measurement runner so each frozen microbatch can finish
prefill in one scheduler event; otherwise the pure-decode recorder sees the
mixed prefill/decode transition that consumed V4. Chunking is still available
for batches that exceed the budget.

The repair closes the ingress geometry but changes the engine's maximum input
and compilation envelope. Under that envelope, the measured KV capacity falls
from V4's 392,432 tokens (24,527 blocks) to 318,848 tokens (19,928 blocks), a
loss of 4,599 blocks. This is not merely a loss of the 682-block conservative
safety margin: measured capacity is also below the frozen 21,000-block live-KV
requirement.

## Fail-closed disposition

The probe wrote its result before rejecting the insufficient capacity. The
parent stopped on the non-zero child result. It did not issue a prompt,
generate a token, retry, use a fallback GPU, create a value-screen output, or
grant performance or action authority.

The create-only result consumes
`data/p4/p4_b0_full_prefill_resource_probe_authorization.json`. That historical
authorization must not be reused, and its source closure is intentionally
invalidated by the post-result historical test update.

## Validation

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  research/97_composition_runtime/tests/test_p4_b0_capture_runner.py \
  research/97_composition_runtime/tests/test_p4_b0_full_prefill_resource_probe.py \
  -q

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  research/97_composition_runtime/tests -q
```

The focused runner/probe suite passes 25 tests. The complete Phase 97 suite
passes 299 tests and 21 subtests. Ruff format and check pass all touched Phase
97 Python files. GPU 4 has no remaining compute process.

## Next decision

Do not prepare V5 from this result. A separate review must choose and justify
a resource repair that preserves the frozen prompt bundle, pure-decode
accounting, shared target KV, and at least 21,682 measured blocks. Candidate
engine-memory or graph-envelope changes require a new source-bound,
one-initialization GPU-4 probe authorization and a fresh create-only result.
Reducing the 21,000-block workload requirement is a workload redesign, not a
measurement repair, and requires new preregistration.

No GPU command is currently authorized.
