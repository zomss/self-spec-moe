# Phase 97 P4 — chunked-prefill probe attempt V4

Date: 2026-08-09

Status: **STOPPED after the one authorized GPU-4 boot; all three cohort
barriers completed, but the probe-only final validator rejected valid
randomized request IDs; V4 is consumed and no probe result or score exists**

## Outcome

The exact V4 command was executed once. Parent preflights passed, and the
compiled target-matching K4 engine booted on physical GPU 4 with configured
and effective budgets `8192 / 8160`, 32 sequence slots, a 0.55 GiB actual CUDA
graph pool, and 24,527 shared target-KV blocks.

The phase-aware V3 repair worked on the live path. Thirty-nine model steps
processed R4, R5, and R5cot chunked prefills, including wide prefill-side
draft queries. Each cohort then released all eight requests in one pure-decode
event at steps 8, 23, and 38. Every release used K4, draft step-0 query width
one, draft output width four, zero preemption, zero recomputation, and exact
same-event closure. The child held three complete cohort histories and three
same-event records before final validation.

The final probe validator failed on R4 request-row identity. vLLM had preserved
its normal randomized internal IDs, for example `R4-s0-p000-a3388847`, while
the frozen identity is `R4-s0-p000`. Every observed suffix was exactly eight
lowercase hexadecimal characters, and canonicalization maps the observed set
exactly to the frozen eight-request set.

The ordinary `P4SameEventRecorder` and the cohort barrier already call
`canonicalize_p4_request_ids`. The probe-specific `_NonScoredRecorder` stores
the raw event, and `_validate_probe_evidence` compared those raw IDs directly
with frozen IDs. This probe-only representation omission caused
`probe cohort R4 target rows changed identity` after the substantive GPU work
had completed.

V4 was not retried or resumed. No fallback GPU, scoring, V10, P4a work, action
admission, or performance claim is authorized.

## What passed

- GPU-4 identity and idle-state, native sampler, and in-process EngineCore
  preflights;
- compilation, CUDA graph capture, and engine initialization;
- 291 target/draft weight aliases and all 36 draft attention layers bound to
  the target-owned KV cache, with no private draft KV allocation;
- 392,432 target-KV tokens, or 24,527 blocks, above the 21,682-block floor;
- the configured/effective `8192 / 8160` scheduler contract;
- phase-aware wide-query chunked prefill across all three regimes;
- atomic width-one K4 decode releases for all 24 requests; and
- zero preemption, zero recomputation, and token closure on all 39 traced
  engine steps.

These are diagnostic partial results. The child emitted no
`probe_result.json`, so the formal probe did not pass.

## Test gap and required repair

CPU coverage exercised canonical frozen IDs or the ordinary recorder's
existing canonicalization. It did not pass valid randomized internal IDs
through the probe-only final validator.

The narrow next step is to:

1. canonicalize probe event request rows with the existing shared helper
   before frozen-identity comparison;
2. retain internal-ID randomization and fail closed on malformed suffixes,
   foreign IDs, and canonical collisions;
3. add direct positive and negative probe-result regressions; and
4. pass the complete Phase 97 CPU surface before considering a separate,
   source-bound V5 authorization at a new create-only output.

This report does not authorize V5 or any GPU execution.

The narrow repair and complete CPU review later passed. A separate source-bound
V5 authorization at a new absent output is documented in
`results_p4_b0_chunked_prefill_probe_authorization_v5.md`; this consumed V4
output remains untouched.

## Immutable artifacts

- failure record:
  `data/p4/run_b0_chunked_prefill_probe_v4/failure.json`
  (`a12bb41ddc31ceb96f35fad9e750d95908045bc6d4b7ff41634c2202894fde0e`);
- child log:
  `data/p4/run_b0_chunked_prefill_probe_v4/child.log`
  (`70572141de9f7d7eedefe9ecf1fc21ac380bb8cdf7e2791adabf59f8939f0f9f`);
- 40-record K/OFF trace:
  `data/p4/run_b0_chunked_prefill_probe_v4/koff_trace.jsonl`
  (`962b943b3a45b9f90d7130be757121c76cc3ccc82a1c165f336c94be745123b7`);
  and
- preparation:
  `data/p4/run_b0_chunked_prefill_probe_v4/preparation.json`
  (`fd1bb870c842e7b5d8625139c01768a9e130fd13c411a88361406bf2e5ae6a46`).
