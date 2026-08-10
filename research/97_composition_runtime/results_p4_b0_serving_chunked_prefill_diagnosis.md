# Phase 97 P4 — real-serving chunked-prefill diagnosis

Status: **PASS for the non-scored `8192 / 0.90` serving boundary. Natural
mixed prefill/decode occurred, every mixed decode row was forced to q=1/OFF,
no draft was dispatched, later pure decode completed, and target-owned shared
KV remained intact. This does not repair or score the V5 screen.**

Date: 2026-08-09.

## Contract

The source-bound package authorized one GPU-4 boot with:

```text
max_num_batched_tokens                              8,192
effective scheduler budget                         8,160
chunked prefill                                      true
gpu_memory_utilization                               0.90
prompt workload                     first frozen R4/seed-0 batch
prompt count / tokens                         8 / 65,200
output tokens per request                              16
K schedule                         OFF for all reachable batch sizes
recording                         append-only, non-scored K/OFF trace
P4 capture recorder                                  disabled
```

The runner used the same Qwen3-8B revision and the target-matching draft path
as the screen. The unreachable batch-33 K4 entry retained the same dynamic
K/OFF graph pool while every reachable batch selected OFF.

## Live result

The run produced 24 target-step records:

```text
initial pure-prefill records                            1
mixed prefill/decode records                            8
later pure-decode records                              15
shared target-KV capacity                          24,527 blocks
complete request outputs                         8 x 16 tokens
additional eligibility exclusions                    none
```

The first mixed step occurred immediately after the first prompt completed
prefill:

```text
computed tokens before step                 [8077, 0, 0]
prompt tokens                                [8077, 7839, 8368]
scheduled query widths                          [1, 7839, 320]
verified / next action                              OFF / OFF
selection intent                                    force_off
draft widths                                        [0, 0, 0]
score eligibility                                       false
exclusion                            prefill_or_mixed_batch
```

Across all eight mixed records, completed decode rows had width one, prefill
rows carried the remaining chunks, and no K4 proposal survived or dispatched.
The mixed query-width sequences were:

```text
[1, 7839, 320]
[1, 1, 8048, 110]
[1, 1, 1, 8157]
[1, 1, 1, 168, 7836, 153]
[1, 1, 1, 1, 1, 7723, 432]
[1, 1, 1, 1, 1, 1, 8154]
[1, 1, 1, 1, 1, 1, 11, 8143]
[1, 1, 1, 1, 1, 1, 1, 29]
```

## Invariants

Every record retained:

- all 36 draft attention layers aliasing the target KV tensors;
- one stable target KV pool, binding, and canonical slot identity;
- no private draft-KV allocation;
- all 291 draft parameters aliasing target parameters; and
- exact target/draft weight-version identity.

The 24,527-block capacity is specific to the real 8,192-token compile/input
envelope. It must not be substituted for the lower capacity observed with the
114,688-token measurement envelope.

## Decision

Real chunked prefill at `8192 / 0.90` behaves as designed: mixed target passes
are safe because existing decodes fall back to q=1/OFF, and the system later
returns to pure decode. Mixed records remain unsuitable for the frozen P4
decode-only score. This is a functional correctness result, not a throughput,
latency, action-admission, or production-value claim.

The V5 screen remains failed and unscored. Its first-event exclusion was not
persisted, so the next screen work is a separate non-scoring full-prefill
ingress diagnosis followed by a fresh authorization—not a change to the real
serving `8192 / 0.90` boundary.

## Artifacts

- authorization:
  `data/p4/p4_b0_serving_chunked_prefill_diagnosis_authorization.json`;
- result: `data/p4/run_b0_serving_chunked_prefill_diagnosis_v1/diagnosis.json`;
- raw trace:
  `data/p4/run_b0_serving_chunked_prefill_diagnosis_v1/koff_trace.jsonl`;
- request outputs:
  `data/p4/run_b0_serving_chunked_prefill_diagnosis_v1/request_outputs.json`;
  and
- preparation evidence:
  `data/p4/run_b0_serving_chunked_prefill_diagnosis_v1/preparation.json`.
