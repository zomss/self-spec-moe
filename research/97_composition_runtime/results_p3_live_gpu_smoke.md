# Phase 97 — registered minimal-B0 live GPU correctness smoke

Follow-up: this original gate remains a valid historical failure, but its
mixed-boundary policy was repaired and passed in
`results_p3b_mixed_abort.md` under the frozen target-local contract.

Status: **FAIL. The live structural K/OFF contract passed, but greedy token
identity failed at the first token emitted by the mixed `K4 -> OFF` boundary.
P4 window work remains blocked.**

Date: 2026-08-09.

## Scope

This is one non-scored correctness smoke. It is not a throughput, latency,
capacity, or selector-quality measurement, and it does not enable profiling.

The switched engine runs on GPU 4 and its independent autoregressive oracle
runs on GPU 5. Both use `Qwen/Qwen3-8B`, TP1, greedy decoding, the same prompts,
and the same per-request output lengths. The switched engine uses the exact
minimal-B0 realization: target-matching aliased weights, one target-owned BF16
KV cache, `K_max=4`, compiled target execution, and no window, layer skip,
draft quantization, private KV, or ahead chain.

## Frozen transition

The dynamic schedule is exactly:

| scheduled batch | action |
| ---: | --- |
| 1 | `target-matching-k4` |
| 2 | `off` |

The long request starts alone. A three-token request is injected only after
the trace contains an eligible stable `K4 -> K4` target step. The batch grows
to two, forcing `K4 -> OFF`; when the short request finishes, it shrinks to
one, forcing a pure-decode `OFF -> K4` step. No worker state is mutated by the
harness.

Fixed engine settings are `max_model_len=2048`, `max_num_seqs=2`,
`max_num_batched_tokens=2048`, `gpu_memory_utilization=0.80`, synchronous
scheduling, prefix caching disabled, and default compiled execution
(`enforce_eager=False`). The long and short requests return exactly 40 and 3
tokens with `temperature=0`, `ignore_eos=True`, and seed 0.

## Pass criteria

The run passes only if one checked verifier establishes all of the following:

1. both switched outputs are token-identical to the greedy AR outputs;
2. the trace contains ordered stable-K4, `K4 -> OFF`, `OFF -> K4`, and
   subsequent verified-K4 records;
3. the pure-decode `OFF -> K4` record has target query width 1, draft step-0
   query width 1, and produces four drafts;
4. every action record is a target-step boundary;
5. live target/draft KV tensor aliases, the target KV pool, target slot-buffer
   identity, and target/draft parameter aliases remain exact and stable;
6. both OFF and K4 have replay-eligible records and every H/D/A/C/E record
   closes exactly; and
7. no active draft-private KV pool is possible under the allocation-time
   alias and target-owned-pool checks.

Any engine exception, missing transition, identity change, counter failure, or
token mismatch is a failed correctness gate. Runtime modes are reported as
observed; this smoke does not convert them into a performance claim.

## Commands and artifacts

The run used `.venv/bin/python` through the checked phase-local harness and
verifier. Outputs were written under `data/p3/` and logs under `logs/` with the
`live_smoke_20260809` stem. The trace writer and every generated artifact
refuse overwrite.

## Result

The successful switched boot used repository commit
`cc8ed50f22568a4f6b88b299e9c47ebba7a8046a`, vLLM
`0.1.dev18402+gcc8ed50f2.d20260808`, torch `2.11.0+cu129`, CUDA 12.9, and an
H100 80GB. GPU 4 returned to 0 MiB after shutdown. The matched V1 AR oracle on
GPU 5 used the same checkpoint, compiler mode, prompts, seed, native sampler,
and request-arrival point and also returned to 0 MiB.

The native sampler was selected with `VLLM_USE_FLASHINFER_SAMPLER=0` because
the installed FlashInfer sampling extension was linked against
`libcudart.so.13`, which is unavailable in the CUDA-12.9 torch environment.
This is a correctness-only greedy run and makes no sampler-performance claim.

### Structural checks that passed

The trace contains 12 engine-step records, of which 10 are replay-eligible.
It shows stable K4, `K4 -> OFF` at engine step 4, `OFF -> K4` at engine step 7,
and subsequent verified K4 steps. The re-entry record is pure decode with
target query width 1, draft step-0 query width 1, and four produced drafts.

Allocation-time and per-step validation proved:

- all 36 draft attention layers alias their target KV tensors;
- the target KV binding, pool, and canonical slot-buffer identities stay
  stable for the full run;
- all 291 draft parameters alias their target parameters;
- model loading reports no draft-side KV allocation; and
- every record is attached to a target-step boundary.

Eligible H/D/A/C/E totals are `H=12`, `D=6`, `A=24`, `C=0`, and `E=36`, so
`E + C = A + H` closes exactly. Stable K4 target passes used FULL execution;
OFF and q=1 re-entry passes used PIECEWISE execution. Draft step-0 and chain
passes reported `NONE`. The mixed `K4 -> OFF` pass reported target mode
`NONE` and is correctly excluded from replay as `prefill_or_mixed_batch`.

### Failed token-identity gate

The short request is exactly identical to AR. The long request matches AR for
tokens 0 through 11 and then diverges at zero-based index 12:

| output | token id at index 12 |
| --- | ---: |
| switched minimal B0 | 22406 |
| matched V1 AR | 5005 |

Index 12 is the first token emitted by engine step 4: a mixed
prefill-plus-decode pass that verifies `target-matching-k4`, accepts all four
drafts, commits five long-request tokens, and selects `off` for the following
dispatch. The mismatch therefore begins exactly at the live `K4 -> OFF`
admission boundary, not at the later q=1 `OFF -> K4` bootstrap.

An initial V2 AR comparison also first diverged at index 12, but it is not the
authoritative control. The reported failure uses the rerun with V1 forced and
the short request injected after the same seven committed long-request tokens
as the switched run.

This initial result alone did not identify whether the fault was mixed-batch
target execution, query/slot values within the stable backing buffers, or
provisional shared-KV state. The completed follow-up in
`results_p3_mixed_query_width_diagnosis.md` now rules out malformed packing,
overlapping slots, provisional shared-KV state, and the K4-to-OFF selector.
It localizes the strict-identity failure to BF16 target execution-shape
numerics at zero- or one-step greedy margins.

## Decision and next gate

The original live gate remains failed; its diagnosis does not constitute a
repair. Do not add a window, layer-skip, or quantized-weight action yet. The
next task is to freeze the correctness contract and mixed-boundary policy.
The recommended boundary policy aborts pending K4 drafts on mixed admission,
runs the existing decode requests at q=1/OFF, and validates it against a
same-boot OFF control. A strict independently booted sequential-AR identity
gate additionally requires a query-width-invariant target realization;
`VLLM_BATCH_INVARIANT=1` was not sufficient in this environment.

## Artifacts

- `data/p3/live_smoke_20260809_trace.jsonl`: switched live trace;
- `data/p3/live_smoke_20260809_spec.json`: switched outputs and injection
  point;
- `data/p3/live_smoke_20260809_ar_v1_matched.json`: authoritative AR oracle;
- `data/p3/live_smoke_20260809_verification.json`: checked FAIL report; and
- `logs/live_smoke_20260809_spec_attempt4.log` and
  `logs/live_smoke_20260809_ar_v1_matched.log`: successful-run logs.

Three pre-trace launch attempts are retained as environment diagnostics. The
first failed the checkpoint-string guard after offline ID/path normalization;
the next exposed the missing venv `ninja` path; the third exposed the
FlashInfer CUDA-13 sampling-extension mismatch. None created a live trace or
contributed token evidence.
