# Phase 97 P4 — chunked-prefill probe attempt V3

Date: 2026-08-09

Status: **STOPPED after one GPU-4 engine boot and the first chunked-prefill
model step; V3 is consumed, no cohort completed, and no probe result or score
exists**

## Outcome

The exact V3 command was executed once. All parent preflights passed, and the
child booted the compiled target-matching K4 engine on physical GPU 4. The
repaired resource and budget gates passed with configured/effective budgets
`8192 / 8160`, 32 sequence slots, a 0.55 GiB actual graph pool, and 24,527
shared target-KV blocks.

The first R4 scheduler event then correctly armed K4 during pure chunked
prefill. It contained no decode request: p000 scheduled all 8,077 prompt
tokens and p001 scheduled the remaining 83-token budget, filling the effective
8,160-token bound. The prefill-side draft proposer ran with output width four
and reported step-0 query width 4,081.

`make_runner_evidence` applied the K4 width-one decode invariant solely from
`next_action_id`, even though scheduler metadata explicitly reported
`pure_decode=false` and an empty decode-request set. It therefore rejected the
valid prefill-side execution before the scheduler could consume the output.
The shutdown barrier then independently refused the necessarily incomplete
cohort.

V3 was not retried or resumed. No fallback GPU, V10, scoring, P4a work, action
admission, or performance claim is authorized.

## What passed

- GPU 4 identity and idle-state preflight;
- native sampler and in-process EngineCore preflights;
- the repaired `8192 / 8160 / 32` admission contract;
- 291 target/draft weight aliases;
- all 36 draft attention layers bound to the target-owned KV cache, with no
  private draft KV allocation;
- CUDA graph capture and engine initialization;
- 392,432 target-KV tokens, or 24,527 blocks, above the 21,682-block gate; and
- zero preemption and zero recomputation at the failed scheduler event.

These observations are diagnostic partial evidence only. The child emitted no
`probe_result.json`, and R4, R5, and R5cot produced no complete cohort.

## Test gap

The CPU cohort helper supplied `draft_step0_query_width=1` whenever K4 was
armed, including the pure-prefill arming step. Other runner-evidence tests
covered pure-decode K4 and mixed-abort OFF, but none represented a K4
prefill-side proposer invocation with a non-decode query width. The live
8,077+83 chunk therefore exercised an untested phase distinction.

## Required repair and next gate

The narrow repair should:

1. require exact draft step-0 query width one only for pure-decode K4 events;
2. retain K4 output-width, proposal-called, execution-mode, shared-KV, and
   shared-weight validation during prefill-side dispatch;
3. keep all prefill and mixed events unscored, while retaining the atomic
   pure-decode requirement for the first measured cohort event;
4. add positive non-one-width prefill and negative non-one-width pure-decode
   execution-path regressions; and
5. pass focused and Phase 97 CPU validation before a fresh source-bound V4
   authorization and create-only output are considered.

This document does not authorize V4 or any GPU execution.

The phase-aware guard and paired CPU regressions later passed under the
separate source-bound V4 package documented in
`results_p4_b0_chunked_prefill_probe_authorization_v4.md`. V4 was later
executed once and stopped after its three complete cohorts on a probe-only
request-ID canonicalization omission; see
`results_p4_b0_chunked_prefill_probe_attempt_v4.md`. This consumed V3 output
remains untouched.

## Immutable artifacts

- failure record:
  `data/p4/run_b0_chunked_prefill_probe_v3/failure.json`
  (`7117131be7c0508c1fd454785e1c67a708b1b4a582f6bd0ae6f124bf79c655ab`);
- child log:
  `data/p4/run_b0_chunked_prefill_probe_v3/child.log`
  (`215af247c7e9a0015b2248836a0801df1b4a31fdea54ee3d3ee01ed12e63f1d0`);
- K/OFF trace header:
  `data/p4/run_b0_chunked_prefill_probe_v3/koff_trace.jsonl`
  (`0407fed3e3ef4f25a092a2bba1870362e13f5fc203e2866a1fd49ab71a8c506c`);
  and
- preparation:
  `data/p4/run_b0_chunked_prefill_probe_v3/preparation.json`
  (`c5b830bebc737b3cae78e0e97e51040fab7e26d1bee8f11e1bf7c06f85aa4de4`).
