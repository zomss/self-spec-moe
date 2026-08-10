# Phase 97 P4 — B0 same-event adapter and scorer freeze

Status: **PASS for the CPU measurement interface and scorer contract; BLOCKED
for live capture, GPU measurement, P4a engineering, admission, and production
claims.**

Date: 2026-08-09.

## Decision

The same-event measurement adapter and B0 scorer are now frozen. This clears
`same_event_measurement_adapter_frozen` and `runner_scorer_frozen`, but it does
not clear `same_event_recorder_unwired`: no live-engine code was changed and
no GPU work was run.

The remaining B0 run-readiness blockers are:

1. live same-event recorder wiring;
2. boot-static w512 mask-equivalence proof; and
3. a conservative candidate resource bound over measured B0.

Every authorization remains false.

## Capture and adaptation boundary

One raw capture is exactly one
`(boot block, action, regime, content seed, round)` cell. It binds the frozen
prompt manifest and bundle, the counterbalanced action position, generation
parameters, stable engine configuration, shared-target-KV proof ids, and the
canonical target-step events.

The adapter calls the registered event validator for every event. It rejects
the entire round if any event is incomplete, mixed-prefill, preempted,
recomputed, invalid-spec, a transition, or missing an explicit field. It also
requires:

```text
request ids = exact 32 frozen prompt record ids
each request commits exactly max_output_tokens
E = 32 * max_output_tokens
E + C = A + H
U = H - D, with 0 <= D <= H
decode_rate_req = E / sum(request_decode_time_s)
tau_raw = 1 + A / H
```

No Prometheus delta, finished-request histogram, or inferred request split can
enter the adapted round.

## Legacy P3 disposition

The existing P3 `koff_engine_step` traces remain useful diagnostics, but they
are not score input. In general they contain only aggregate `H/D/A/C/E`, omit
explicit `invalid_spec_tokens`, and do not provide the per-request accepted,
committed, and clipped rows needed to close a multi-request event. Some also
have `verified_action_id != next_action_id`.

The adapter therefore rejects P3 records instead of filling missing values.
Even the subset of singleton events that could be reconstructed is not reused,
because the contract is uniform and missing fields are never inferred.

## Frozen scorer

The input matrix is exact:

```text
3 paired boot blocks
x 3 actions (OFF, target-matching K4, masked w512 K4)
x 6 W3 regimes
x 2 content seeds
x 4 rounds
= 432 adapted round records
```

Episode handling is frozen to W3:

- the reference is the maximum round for batch at least 8 and the
  second-highest round for b1;
- rounds below 95% of the reference are rejected;
- every boot must retain at least two rounds; and
- all three surviving boot rates must agree within 2%, otherwise the complete
  score input fails closed.

Surviving counters and request time are summed within a boot and then across
the two content seeds. Uncertainty uses 4,000 paired complete-boot-block draws
with Python RNG seed `20260809`; one sampled block keeps every action, regime,
and seed together.

For each draw:

```text
S_k4 = rate_k4 / rate_off
S_w512,no-credit = S_k4 * tau_w512 / tau_k4
base = max(1, S_k4)
candidate = max(1, S_k4, S_w512,no-credit)
gain = candidate - base
```

A w512 regime whose lower bound regresses more than 1% against K4 is masked
and retains K4/OFF. The scorer reports the +2% proper-subset benefit bound,
the acceptance-dominance short circuit, the +2% equal-weight mean branch, and
the +5% single-regime branch with nonnegative mean. A score never grants
authority.

## Artifacts

```text
schemas/p4_b0_same_event_capture.schema.json
schemas/p4_b0_adapted_round.schema.json
schemas/p4_b0_runner_scorer.schema.json
data/p4/p4_b0_runner_scorer_contract.json
data/p4/p4_b0_runner_scorer_validation.json
scripts/adapt_p4_b0_same_event.py
scripts/score_p4_b0.py
tests/test_p4_b0_adapter_scorer.py
```

The B0 preregistration now hashes the runner/scorer contract. Its blocker set
contains only live recorder wiring, mask equivalence, and the conservative
resource bound.

## Validation

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research/97_composition_runtime/scripts/score_p4_b0.py \
  --contract \
  research/97_composition_runtime/data/p4/p4_b0_runner_scorer_contract.json

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  research/97_composition_runtime/tests/test_p4_b0_adapter_scorer.py -q
```

The focused adapter/scorer suite passes 21 CPU tests, including explicit
equal-work adaptation, legacy-P3 rejection, transition/incomplete/invalid
event rejection, b1 and batch-at-least-8 episode controls, two-round minimum,
2% cross-boot failure, deterministic paired bootstrap, dominance, and both
non-authorizing value outcomes.

The full Phase 97 CPU suite passes 177 tests plus 14 subtests. Ruff 0.14.0
check and format-check pass all five touched B0 Python files.

## Next step

Wire the live synchronous engine boundary to emit the raw capture contract.
That work must expose per-request accepted, raw, committed, and clipped token
counts plus explicit invalid-spec counts from the same scheduler event. It
must not change the frozen adapter or scorer, run the GPU screen, or authorize
P4a. If the live record cannot supply an explicit field, the capture remains
unscoreable.
