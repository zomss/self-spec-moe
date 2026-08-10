# Phase 97 P3 follow-up — live minimal-B0 K/OFF engine wiring

Status: **IMPLEMENTED, CPU-VALIDATED, AND LIVE-BOOTED. The live vLLM scheduler,
worker, target runner, and target-matching proposer carry the closed
`OFF`/`K=4` contract and emit passive engine-step JSONL records. The first live
smoke passed all structural checks but failed greedy token identity at the
mixed `K4 -> OFF` boundary. This is not a correctness, resource, or performance
claim.**

Date: 2026-08-09.

## Scope

This follow-up wires only the exact P3 minimal B0 realization:

| action id | verified target width | next draft width | KV | weights |
| --- | ---: | ---: | --- | --- |
| `off` | 1 | 0 | target-owned shared KV | target aliases |
| `target-matching-k4` | 5 | 4 | target-owned shared KV | target aliases |

No window, skipped layer, draft-only KV dtype, private KV, partial replica,
ahead chain, consumed-ahead chain, quantized draft-weight path, or other `K`
is admitted. Target quantization remains fixed input and the target and draft
model/quantization configurations must match.

## Live lifecycle

A target pass verifies drafts produced in the previous engine cycle while
`num_spec_tokens_to_schedule` selects the drafts produced after that pass.
The transition record therefore carries two action ids:

```text
draft action from cycle t-1
    -> target pass t verifies verified_action_id
    -> scheduler selects next_action_id
    -> proposer emits drafts for cycle t+1
```

For example, an OFF-to-K4 transition is represented as
`verified_action_id=off` and `next_action_id=target-matching-k4`. Collapsing
these into one id would falsely label the target q=1 pass as a K4 q=5 pass.

The shared-KV step-0 compactor now recognizes the first uniform q=1 target
pass after OFF even though no speculative rejection tensor exists yet. It
compacts the appended-token draft forward to q=1, matching the registered K4
draft path. Steady-state K4 continues to use the rejection-aware q=1 path.

## Engine integration

The implementation adds:

- a closed action registry and validation helpers in
  `vllm/v1/spec_decode/koff_runtime.py`;
- opt-in `VLLM_SELF_SPEC_KOFF_RUNTIME` and create-new
  `VLLM_SELF_SPEC_KOFF_TRACE` controls;
- scheduler metadata on `SchedulerOutput`, worker evidence on
  `ModelRunnerOutput`, and action provenance on `DraftTokenIds`;
- synchronous and asynchronous scheduler provenance handling;
- target/draft query-width and output-width revalidation in the worker;
- post-load target/draft parameter-object alias validation;
- post-allocation draft/target KV tensor-object alias validation;
- a backing-storage fingerprint for the canonical target slot mappings;
- proposer evidence for step-0 query width and draft execution modes; and
- passive per-engine-step H/D/A/C/E, clipping, KV use, preemption, and
  recomputation records.

The slot-map fingerprint ignores transient view shape, layer-dict wrapping,
and DBO list wrapping. It changes when the underlying target slot buffer is
replaced by another allocation.

## Fail-closed boundary

Startup or execution raises `KOffRuntimeError` when it cannot prove the
minimal-B0 contract. The checks include:

1. speculative method is `draft_model` with exactly four speculative tokens;
2. padded draft batches, shared KV, shared-KV step-0 decode, and weight
   sharing are enabled;
3. target and draft checkpoint and quantization identifiers match;
4. dynamic schedules and compiled policies contain only `K=0` or `K=4`;
5. excluded window, skip, private-KV, partial-replica, and ahead-chain
   mechanisms are absent;
6. every live draft KV tensor is the exact target tensor object;
7. every draft parameter is the exact target parameter object;
8. scheduler action provenance, target query width, worker action, produced
   draft width, and proposer step-0 width agree; and
9. shared-KV binding, pool, and canonical slot-buffer identities remain
   stable.

Prefill or mixed steps force the next action to OFF and remain visible in the
trace, but are marked ineligible for P3 replay. Preemption, recomputation,
invalid speculative tokens, missing decode output, and accounting failure
also make a record replay-ineligible.

## Activation contract

Add the following to an existing target-matching B0 launch:

```bash
VLLM_SELF_SPEC_SHARED_KV=1 \
VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE=1 \
VLLM_SELF_SPEC_SHARE_WEIGHTS=1 \
VLLM_SELF_SPEC_KOFF_RUNTIME=1 \
VLLM_SELF_SPEC_KOFF_TRACE=/existing/parent/unique-live-koff.jsonl \
<existing vLLM launch with draft_model, draft=target, and K=4>
```

`VLLM_SELF_SPEC_KOFF_TRACE` is optional. Its parent directory must already
exist and the file must not exist; the writer refuses to overwrite it. The
runtime flag without the three explicit shared-B0 flags fails closed rather
than silently changing the requested boot.

The current K/OFF choice comes from the existing dynamic-K, compiled-policy,
or acceptance-gate path. This wiring does not yet replace that selector with
the Phase 96 interval policy.

## Verification

Commands run without GPU execution:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  tests/v1/spec_decode/test_koff_runtime.py -q

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  tests/v1/core/test_async_scheduler.py -q

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  tests/v1/spec_decode/test_dynamic_sd.py -q

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  tests/v1/core/test_scheduler.py -q \
  -k 'schedule_spec_decoding_stats or no_spec_tokens_scheduled_for_prefill_chunks or preempt_during_execution or abort_request_when_structured_output_fsm_cannot_advance or async_scheduling_pp_allows_rescheduling_with_output_placeholders'

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover \
  -s research/97_composition_runtime/tests -p 'test_*.py' -v
```

The final expected counts are 28 live-contract tests, 9 asynchronous
scheduler tests, 16 dynamic-K tests, 10 focused scheduler tests, and all 50
Phase 97 phase-local tests.

Focused Ruff checks pass for the new runtime module, transport files, and
tests. A whole-file check of the three large integration files still reports
the same five legacy findings present in `HEAD`; no unrelated formatter or
lint churn is included.

## Limitations and next gate

- A non-scored model boot and correctness smoke was run; no profiler or scored
  measurement was run. See `results_p3_live_gpu_smoke.md`.
- The recorded graph ids are registered logical descriptors tied to validated
  query widths. The smoke observed FULL target execution for stable K4,
  PIECEWISE for OFF/q=1, and NONE for the mixed boundary; draft execution was
  NONE. A logical graph id therefore does not by itself prove captured-graph
  dispatch.
- The JSONL output is passive and `scored=false`; it does not construct or
  calibrate Phase 96 intervals.
- The live JSONL is an engine-step stream, while the current phase-local
  replayer consumes aggregated synthetic segments. A checked converter and
  live-record schema are still required before calling the stream replayable.
- One action applies to the whole decode dispatch. A heterogeneous draft-width
  batch fails closed; cohort partitioning is not implemented.
- Multi-DP trace-file routing and pipeline-parallel live behavior have not
  been exercised.
- Exact post-capture HBM and shared-KV block capacity remain the P2 resource
  gate and are not inferred from this code path.

The follow-up diagnosis in `results_p3_mixed_query_width_diagnosis.md` proved
the mixed query and slot packing correct and localized the mismatch to BF16
target execution-shape numerics near greedy ties. Before a window or
layer-skip action enters the live registry, freeze the target-local versus
strict sequential-AR correctness contract and implement the registered
mixed-boundary policy. The recommended policy aborts pending K4 drafts and
runs q=1/OFF on admission. The rerun must retain exact aliases, q=1 bootstrap,
same-boot fixed-action equivalence, and H/D/A/C/E closure. Then add the checked
live-stream converter. A performance claim requires a separate registered
measurement after that correctness gate.
