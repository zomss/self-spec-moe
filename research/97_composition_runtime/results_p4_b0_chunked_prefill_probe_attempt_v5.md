# Phase 97 P4 — chunked-prefill probe attempt V5

Date: 2026-08-10

Status: **GPU PASS; the single authorized non-scored GPU-4 boot completed all
three chunked-prefill cohort barriers and emitted the formal V5 probe result;
V5 is consumed and V10 remains unauthorized**

## Outcome

The exact V5 command was executed once after its source-bound validator,
prepare-only gate, create-only output check, and physical GPU-4 idle check all
passed. The compiled target-matching K4 engine booted on physical GPU 4 with
configured/effective budgets `8192 / 8160`, 32 sequence slots, and the
target-owned shared KV cache.

R4, R5, and R5cot each completed one eight-request cohort. Their atomic first
measured decode events occurred at engine steps 8, 23, and 38. Every event was
pure decode, used K4, processed all eight requests, committed one measured
token per request, and satisfied same-event counter closure. All 24 result
rows use the canonical frozen request IDs; the valid randomized internal IDs
were preserved during execution and normalized only at the evidence boundary.

The run emitted `probe_result.json` with `status=pass`, `scored=false`, and
`authorization_consumed=true`. Individual same-event records satisfy the
score-eligibility shape, but this probe was explicitly non-scored and emitted
no value score. V10, P4a engineering, action admission, and performance claims
remain false.

## Live evidence

- 291 draft parameters aliased the target weights, and all 36 draft attention
  layers bound to the target-owned KV cache with no private draft KV;
- the shared target-KV capacity was 24,527 blocks against the 21,682-block
  minimum;
- actual CUDA graph memory was 589,365,248 bytes, versus a 507,510,784-byte
  estimate;
- the R4/R5/R5cot cohorts completed after 8/14/14 prefill steps respectively,
  followed by one atomic measured decode step each;
- each measured event had `A=C=32`, `D=E=H=8`, `U=0`, and both closure
  predicates true; and
- invalid speculative tokens, preemptions, and recomputed tokens were zero in
  all three measured events.

The append-only K/OFF trace contains 40 records. Independent post-run checks
confirmed the bound GPU UUID, budgets, resource floor, three complete
eight-request cohorts, 24 unique canonical request IDs, zero quality
violations, and the absence of score or downstream authority.

## Decision boundary

This PASS closes only the real `8192 / 8160 / 0.90` chunked-prefill cohort
probe. The next permitted artifact is a separate source-bound V10
authorization review that binds this immutable result. This report does not
authorize V10 execution, scoring, retry, resume, fallback GPU, P4a work,
action admission, or a performance claim.

## Immutable artifacts

- probe result:
  `data/p4/run_b0_chunked_prefill_probe_v5/probe_result.json`
  (`dc398883c6ed62a81ee46ff5643a68216954983d5b7787951218463149120cc2`);
- 40-record K/OFF trace:
  `data/p4/run_b0_chunked_prefill_probe_v5/koff_trace.jsonl`
  (`2dd4e36766e9ce19db7914bfcffd5fa32d2f80744d8df61afc4cc4938e61aa57`);
- child log:
  `data/p4/run_b0_chunked_prefill_probe_v5/child.log`
  (`18998a830781c88afacdde56c8cbc8a113a94888eae691bed85d59ade4f78458`);
- preparation:
  `data/p4/run_b0_chunked_prefill_probe_v5/preparation.json`
  (`2abd491dda5c6dfeabf37c0e98798baf9ea3885552385f72b5267571e16ee13c`);
  and
- consumed authorization:
  `data/p4/p4_b0_chunked_prefill_probe_authorization_v5.json`
  (`795ead82642faaed18f59e759f870aed86dd4c769d5bd4d80344b4003c1d8599`).
