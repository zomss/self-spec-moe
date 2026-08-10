# Phase 97 P4 — bounded chunked-prefill cohort-barrier design and CPU proof

Status: **PASS for the CPU-only design. Bounded prefill, early-release
prevention, exact decode accounting, abort behavior, and OFF/K4/w512 rollover
are proven. The barrier is not wired into the live scheduler, no GPU probe is
authorized, and V10 remains unauthorized.**

Date: 2026-08-09.

## Decision

Replace the rejected `114688 / 0.96` full-microbatch mechanism with bounded
chunked prefill at `max_num_batched_tokens=8192`. With the 32-token
speculative slot reserve, the effective prompt budget is 8,160 tokens. The
frozen R5 and R5cot maxima therefore require at least 14 scheduler events,
rather than one 112k-token compiled forward.

This is the right ingress mechanism, but it is not yet a memory-fit result.
GPU memory utilization remains deliberately unset by this CPU artifact. A
later probe must measure the actual CUDA graph pool and retain at least 21,682
target-owned KV blocks.

The capture-only state machine is:

```text
register every exact member -> seal -> bounded prompt chunks
                                      |
                         one request finishes prefill
                                      |
                         hold its next decode schedule
                         retain its target-owned KV
                                      |
                          every member is now ready
                                      |
                    release all on the next scheduler boundary
                                      |
                       first measured event is pure decode
                                      |
                   exact S_dec work -> frontend offset check
                                      |
                                  complete

any identity, progress, quality, or accounting fault -> abort whole cohort
```

“Hold” does not finish the request, free its blocks, or rebuild its prompt. It
only suppresses decode scheduling after the request has produced its one
unmeasured prefill sample. The target's single shared KV allocation remains
live throughout; private draft KV is forbidden.

## Exact contract

The active capture-cell prompt slice defines one cohort. All members must be
queued in frozen order before the first engine step. Runtime request IDs may be
either the exact frozen ID or that ID plus exactly eight lowercase hexadecimal
characters. Missing, duplicate, reordered, or foreign IDs abort the cohort.

A request becomes held only when both conditions are true:

```text
num_computed_tokens == num_prompt_tokens
num_output_tokens == 1
```

The barrier releases only when every member satisfies that predicate. Release
occurs on the next scheduler boundary and must schedule every unfinished
member in one pure-decode event. The verified target query width must be 1 for
OFF and 5 for K4/w512. Mixed prefill/decode, preemption, recomputation, invalid
speculative tokens, partial release, and action drift all abort without a
capture or score.

The proof uses one unmeasured prefill token plus exactly 512 measured `S_dec`
commits per request, so the frontend must return exactly 513 tokens. The class
is parameterized by the active cell's `generation.max_output_tokens`; the
512-token case proves the offset and rollover mechanism without changing the
other frozen regime work definitions.

## CPU proof

Three positive cohorts pass independently:

| action | target width | prefill events | measured decode events | result |
| --- | ---: | ---: | ---: | --- |
| OFF | 1 | 3 | 512 | exact 1 + 512 closure |
| K4 | 5 | 3 | 103 | exact 1 + 512 closure |
| w512 | 5 | 3 | 103 | exact 1 + 512 closure |

Each positive case uses unequal prompt progress, proves that early members are
held, releases all three members together, and creates a fresh barrier for the
next action. No state crosses the OFF/K4/w512 rollover.

Ten negative cases reject missing membership, unknown identity, early decode,
a second prefill sample, partial release, mixed release, quality corruption,
decode overcommit, two no-progress steps, and premature frontend completion.
A separate member-abort case returns all unfinished cohort IDs for cancellation
and permits neither a partial capture nor scoring.

Ordinary serving is unchanged. It retains the already validated `8192 / 0.90`
boundary, does not instantiate this barrier, and continues to force mixed
prefill/decode events to q=1/OFF with no draft dispatch.

## Artifacts

```text
vllm/v1/spec_decode/koff_runtime.py
research/97_composition_runtime/data/p4/
  p4_b0_chunked_prefill_cohort_barrier_design_and_cpu_proof.json
  p4_b0_chunked_prefill_cohort_barrier_validation.json
research/97_composition_runtime/scripts/
  validate_p4_b0_chunked_prefill_cohort_barrier.py
research/97_composition_runtime/tests/
  test_p4_b0_chunked_prefill_cohort_barrier.py
```

The machine-readable proof source-binds the V9 authorization and failure, the
transient bound, frozen prompt manifest, serving diagnosis, executable state
machine, validator, and tests. V9 and
`data/p4/run_b0_value_screen_v8` remain consumed and immutable.

## Validation

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  research/97_composition_runtime/tests/\
test_p4_b0_chunked_prefill_cohort_barrier.py -q

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research/97_composition_runtime/scripts/\
validate_p4_b0_chunked_prefill_cohort_barrier.py \
  --artifact \
  research/97_composition_runtime/data/p4/\
p4_b0_chunked_prefill_cohort_barrier_design_and_cpu_proof.json
```

The focused suite passes 19 tests and two subtests. The validator reproduces
three positive, ten rejection, and one cohort-abort proof case. The active
proof was revalidated after the phase-aware draft query-width repair; its
state-machine result and authority boundary are unchanged. Ruff check and
format pass the implementation, validator, and tests. The complete Phase 97
audit now reports 474 passed and 43 subtests passed; its only two failures are
the untouched historical V9 run-readiness assertions. The live runner
correctly rejects frozen V9 on source drift before reaching its already
consumed create-new output check. V9 was not refreshed or retried. No GPU
command ran.

## Next step

The synchronous live-engine wiring and its execution-path CPU regressions are
complete. The separate V1 probe authorization was later invoked once and
consumed by a parent relative-path failure before child launch or GPU model
execution.

The path-normalization repair and source-bound V2 package passed CPU review,
but its later one-shot execution exposed a distinct live-guard bug. The
configured batch bound remained 8,192 while the designed speculative reserve
correctly reduced the effective scheduler budget to 8,160; the guard compared
the latter with 8,192 and rejected the first request after engine boot. V2 is
consumed with no cohort or result. The configured/effective repair and real
draft-model normalization regressions now pass, and a fresh source-bound V3
package authorizes one unexecuted non-scored GPU-4 probe at an absent output.
V10 remains unauthorized.
