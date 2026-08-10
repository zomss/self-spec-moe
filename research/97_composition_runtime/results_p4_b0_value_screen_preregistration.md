# Phase 97 P4 — matched B0 value-screen preregistration

Status: **PASS as a pre-measurement registration; BLOCKED for GPU execution,
P4a engineering, admission, and production-value claims. The research
objective, equal W3-suite weights, exact OFF/K4/w512 action set, same-event
accounting semantics, decision thresholds, and the exact 384-prompt token
bundle are frozen. The strict CPU adapter and scorer are now frozen; three
run-readiness artifacts remain missing.**

Date: 2026-08-09.

## Decision

The next P4 experiment is now defined, but it is not authorized to run. It is
a research-only screen over the fixed W3 regime suite, with equal weight on
`R4`, `R5`, `R5cot`, `R8`, `R1`, and `R6`. No measured production workload
distribution was found, so these weights must not be described as rollout or
serving traffic weights.

The base and candidate pools are:

```text
base      = {off, target-matching-k4}
candidate = {off, target-matching-k4,
             target-matching-w512-masked-k4}
```

All three actions retain one target-owned KV pool. Draft-KV quantization,
private KV, layer skip, and quantized draft weights are outside this screen.

## Same-event accounting repair

The binding counter unit is one decode request-step inside one scheduler
engine event. `H`, `D`, `A`, `C`, and `E` must come from the same scheduler
output, model-runner output, and post-stop commit boundary:

```text
H = number of target-processed decode rows
D = armed rows within those same H rows
A = accepted draft tokens on those rows
C = generated tokens clipped before commit
E = committed tokens
U = H - D
```

Every complete event must satisfy:

```text
0 <= D <= H
E + C = A + H
U = H - D
request_decode_time = engine_event_elapsed * H
```

The event schema additionally closes every request row: raw output is
accepted draft tokens plus one target token, committed plus clipped tokens
equals raw output, unarmed rows accept no drafts, and accepted drafts cannot
exceed the registered `K`. Scored records must also be steady state:
`action_id = verified_action_id = next_action_id`; bootstrap and transition
events do not enter action value. Decode time comes from that same scheduler
event, excluding queue and prefill time. Incomplete, mixed-prefill, preempted,
recomputed, or invalid-spec-token events may be retained for diagnostics but
cannot be scored. Prometheus interval deltas and finished-request histograms
are diagnostic-only.

This disposes of the current W14/D bundle without rewriting Phase 96. Its 21
files and 1,680 rounds remain immutable, but 313 rounds have `H-D<0`, with a
minimum of `-8`. The one-batch (`b8`) boundary-skew explanation is plausible
but unconfirmed. The bundle is therefore `invalid_for_scored_value` and
cannot be rescored into this screen.

## Window treatment and the dominance check

The masked w512 action remains `acceptance_only`. A boot-static w512 run may
supply acceptance only after proving that its mask computation, target-matching
weights, shared-KV bindings, and true slot mapping are equivalent to the
proposed action. Its separately booted latency is diagnostic and receives no
value credit.

The conservative candidate score reuses K4 cost:

```text
tau_a = 1 + A_a / H_a
S_w512,no-credit = S_k4 * tau_w512 / tau_k4
base_g = max(1, S_k4,g)
candidate_g = max(1, S_k4,g, S_w512,no-credit,g)
```

This exposes the key risk in the current P4 direction: target-matching K4 has
already shown full `4/4` acceptance in the non-scored P3 traces. A masked
window with no cost credit has little room to add value. The preregistration
therefore stops P4 immediately if every regime has
`UCB(tau_w512) <= LCB(tau_k4)`. Passing that dominance check is necessary but
does not itself authorize engineering.

## Measurement and value rules

- Currency: W3 `S_dec`, per-request decode-rate ratio.
- Work: fixed output length with EOS ignored and identical prompt token ids,
  generation seeds, target/draft versions, batch, suffix, K, graph grade,
  kernels, warmup, and hardware across actions.
- Replication: three counterbalanced paired boot blocks and four rounds per
  boot.
- Noise protocol: W3's 5% episode rule, at least two surviving rounds, the
  b1 second-highest reference, and 2% cross-boot certification.
- Uncertainty: 4,000-draw paired complete-boot-block bootstrap at 95%.
- Missing evidence: `null`, never zero.

The pre-engineering value branch uses lower confidence bounds and passes only
if either:

1. equal-weight mean candidate-pool gain is at least +2%; or
2. one regime gains at least +5% while the equal-weight mean is nonnegative.

The masked action also retains the 1% proper-subset non-regression tolerance;
a failing action is masked while K4 and OFF remain available. Even a value
pass grants no authority until the conservative pre-engineering resource gate
is repaired.

## Exact prompt and tokenizer freeze

The six-regime input is now executable without retokenizing strings. The
bundle contains 384 canonical JSONL records: 32 prompts for each of content
seeds 0 and 1 in each regime. It contains 2,351,468 prompt tokens, uses
generation seed 0 for every matched action, fixes `ignore_eos=true`, and keeps
every prompt-plus-output length within the registered 20,480-token limit.

The manifest pins:

- Qwen3-8B tokenizer revision
  `b968826d9c46dd6066d109eabc6255188de91218` and the four tokenizer-file
  hashes;
- the canonical regime-loader hash and exact generator hash;
- the GSM8K, AIME 1983–2024, CNN/DailyMail, NQ-open, and C4 revisions,
  fingerprints where applicable, and exact cache-file hashes; and
- every prompt's token count and SHA-256 over little-endian uint32 token ids.

The gzip bundle uses `mtime=0`, has SHA-256
`0615cf0174bc476dcc744cb947440b09be4b61043685a262e62f573821de1506`,
and was regenerated byte-for-byte with the same manifest hash
`10de0f73897c08b278d532a5764d3e45ccb67dbc52c1dc9d1e85a3da4c54ad93`.
Freezing this artifact satisfies `exact_prompt_manifest_frozen`; the later CPU
adapter/scorer freeze is documented in `results_p4_b0_adapter_scorer.md`.
Neither authorizes a GPU run.

## Why execution is still blocked

Three artifacts are still required before a GPU command can be registered:

1. live-engine recorder wiring that emits explicit canonical request rows into
   the frozen capture adapter;
2. a w512 mask-equivalence proof for the boot-static acceptance surrogate; and
3. a conservative candidate resource bound on top of the measured B0 base.

Historical P3 traces do not satisfy item 1. They omit general per-request
accept/commit/clipping rows and explicit invalid-spec counts, and some records
are transitions. The adapter rejects those missing fields rather than
reconstructing them from aggregate `H/D/A/C/E`.

The current resource artifact remains a `static_proxy_projection` with
`candidate_realization_match=false`; its relation is
`optimistic_proxy_ceiling`, so its decision remains reject. Transition and
probe overhead remain a post-engineering gate: p95 switch time below one
target step and amortized overhead below 0.5%. If P4a is later implemented,
`measured_exact`/`exact_candidate` resource evidence is still mandatory before
admission; it is deliberately not a pre-P4a requirement.

## Artifacts and validation

```text
schemas/p4_target_step_event.schema.json
schemas/p4_same_event_accounting.schema.json
schemas/p4_b0_value_screen.schema.json
schemas/p4_prompt_manifest.schema.json
schemas/p4_b0_same_event_capture.schema.json
schemas/p4_b0_adapted_round.schema.json
schemas/p4_b0_runner_scorer.schema.json
data/p4/p4_same_event_accounting_contract.json
data/p4/p4_same_event_accounting_validation.json
data/p4/p4_b0_prompt_manifest.json
data/p4/p4_b0_prompt_manifest_validation.json
data/p4/p4_b0_prompt_tokens.jsonl.gz
data/p4/p4_b0_value_screen_prereg.json
data/p4/p4_b0_value_screen_validation.json
data/p4/p4_b0_runner_scorer_contract.json
data/p4/p4_b0_runner_scorer_validation.json
scripts/generate_p4_prompt_manifest.py
scripts/validate_p4_prompt_manifest.py
scripts/validate_p4_same_event_accounting.py
scripts/validate_p4_b0_value_screen.py
scripts/adapt_p4_b0_same_event.py
scripts/score_p4_b0.py
tests/test_p4_prompt_manifest.py
tests/test_p4_same_event_accounting.py
tests/test_p4_b0_value_screen.py
tests/test_p4_b0_adapter_scorer.py
```

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research/97_composition_runtime/scripts/validate_p4_same_event_accounting.py \
  --contract \
  research/97_composition_runtime/data/p4/p4_same_event_accounting_contract.json

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research/97_composition_runtime/scripts/validate_p4_prompt_manifest.py \
  --manifest \
  research/97_composition_runtime/data/p4/p4_b0_prompt_manifest.json

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research/97_composition_runtime/scripts/validate_p4_b0_value_screen.py \
  --preregistration \
  research/97_composition_runtime/data/p4/p4_b0_value_screen_prereg.json

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research/97_composition_runtime/scripts/score_p4_b0.py \
  --contract \
  research/97_composition_runtime/data/p4/p4_b0_runner_scorer_contract.json
```

The focused prompt, accounting, value-screen, and adapter/scorer suites contain
91 CPU tests. The full Phase 97 suite passes all 177 tests plus 14 subtests.
Ruff 0.14.0 check and format-check pass the five touched B0 Python files.

## Next artifact

Build the remaining run-ready package in this order:

1. wire the live engine to the frozen capture schema and prove the adapter can
   consume it without inferred fields;
2. prove boot-static w512 acceptance equivalence; and
3. produce a conservative pre-engineering resource bound.

Only then may a separate decision authorize the matched GPU screen. If the
dominance short-circuit fires, stop P4 and retain the executable K4/OFF pool;
do not implement the masked action. A later P4a must still produce exact
post-capture resource evidence before admission. P5 and B1 remain
independently gated.
