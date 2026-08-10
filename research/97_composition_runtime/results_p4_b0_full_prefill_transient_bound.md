# Phase 97 P4 — B0 full-prefill transient-memory bound

Status: **FAIL for the current `114688 / 0.96` full-microbatch geometry. It
cannot retain the registered KV floor and safely execute R5/R5cot under the
observed graph/allocator state. This result grants no GPU or V10 authority.**

Date: 2026-08-09.

## Decision

The V9 OOM closes the current repair direction. Two independent conservative
checks reject a same-geometry V10:

1. Correcting only the minimum graph-pool underestimate permitted by the
   two-decimal reports reduces the optimistic KV capacity from 22,090 to
   20,092 blocks, below the 21,000-block hard live-workload requirement.
2. Even if the graph state is held fixed and all 408 blocks above the 21,682
   launch floor are released, the first failed R5 allocation remains at least
   235,095,982 bytes short under optimistic rounding.

Therefore lowering `gpu_memory_utilization` enough to cover this allocation
would violate the registered floor, while retaining the floor reproduces the
OOM risk. Retry, resume, GPU 5, and trust in the graph estimate are not valid
repairs.

## Graph-pool correction

The launch reported 0.50 GiB estimated and 4.90 GiB actual CUDA graph-pool
memory. Treating both as nearest-rounded two-decimal values gives a minimum
possible underestimate of 4.39 GiB:

```text
minimum graph miss                              4,713,726,608 bytes
KV block bytes                                      2,359,296 bytes
minimum graph miss in KV blocks                         1,998
observed KV blocks                                     22,090
optimistic graph-corrected blocks                       20,092
hard live-workload requirement                          21,000
shortfall                                                  908
```

This correction does not claim that graph memory literally belongs in the KV
pool. It expresses the amount by which pre-KV memory planning understated a
post-capture resident cost, using KV blocks as the common HBM currency.

## Failed-allocation floor check

The observed pool had only 408 blocks above the launch floor. Releasing every
one would recover 962,592,768 bytes. The allocator reported 1.45 GiB free; an
optimistic upper bound under nearest two-decimal rounding is 1.455 GiB, or
1,562,294,354 bytes:

```text
optimistic free plus all floor headroom          2,524,887,122 bytes
failed R5 SiLU/multiply request                   2,759,983,104 bytes
minimum remaining deficit                          235,095,982 bytes
additional KV-block equivalent                              100 blocks
maximum compatible KV capacity                           21,582 blocks
registered launch floor                                  21,682 blocks
```

This check concerns only the first allocation that failed. It gives no credit
for fragmentation and does not assume later layers or R5cot would fit.

## Static R5/R5cot live-set lower bound

Qwen3-8B has hidden size 4,096 and intermediate size 12,288 at TP1. At the
compiled SiLU/multiply output allocation, the following bfloat16 tensors are
simultaneously live:

| tensor | width per prompt token |
| --- | ---: |
| attention output projection | 4,096 |
| post-attention normalized hidden | 4,096 |
| residual | 4,096 |
| merged gate/up projection | 24,576 |
| SiLU/multiply output | 12,288 |
| **total** | **49,152** |

At two bytes per element, this is 98,304 bytes per prompt token:

| regime | prompt tokens | gate/up | SiLU output | total live-set lower bound |
| --- | ---: | ---: | ---: | ---: |
| R5 | 112,304 | 5.141 GiB | 2.570 GiB | 10.282 GiB |
| R5cot | 112,908 | 5.169 GiB | 2.584 GiB | 10.337 GiB |

This is a model-activation lower bound, not a complete allocator peak. It
excludes attention workspace, allocator fragmentation, and other runtime
objects. It must not be subtracted from KV capacity again on top of the
observed OOM or graph correction.

## Better direction

The next engineering step should restore bounded chunked prefill and add a
measurement-only capture-cohort decode barrier:

1. queue all frozen requests before the first engine step;
2. permit bounded prompt chunks while holding any request that has finished
   prefill after its unmeasured prefill sample;
3. release the complete cohort together into the first measured pure-decode
   step;
4. retain exactly 512 measured `S_dec` commits per request and the existing
   same-event accounting; and
5. leave real serving behavior unchanged—mixed serving steps continue to
   force q=1/OFF as already validated at budget 8192.

This directly removes the 112k-token compiled forward without weakening the
shared-KV or fixed-work contracts. It is preferable to implementing new
low-memory kernels because serving-stack optimization is outside this
research's main scope.

Before V10, the barrier requires a CPU fail-closed state-machine proof and a
separately authorized, non-scored one-boot GPU probe. That probe must bind
actual graph-pool memory, at least 21,682 post-capture target-KV blocks, pure
first measured decode for R4/R5/R5cot, and zero preemption, recomputation, or
invalid speculative tokens.

## Validation

The machine-readable bound is
`data/p4/p4_b0_full_prefill_transient_bound.json`. It binds the V9
authorization/failure/output, frozen prompt manifest, exact model config, and
checked-in Qwen MLP sources. It explicitly excludes TorchInductor cache files
from required evidence. The validation summary is
`data/p4/p4_b0_full_prefill_transient_bound_validation.json`.

Validate with:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research/97_composition_runtime/scripts/\
validate_p4_b0_full_prefill_transient_bound.py \
  --bound \
  research/97_composition_runtime/data/p4/\
p4_b0_full_prefill_transient_bound.json

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  research/97_composition_runtime/tests/\
test_p4_b0_full_prefill_transient_bound.py -q
```

The focused suite passes 13 tests. Ruff check and format pass the validator
and tests. The complete Phase 97 suite reports 427 passed, 41 subtests passed,
and two failures in the historical V9 pre-execution tests: both deliberately
require the create-new V9 output to be absent and now fail closed because the
consumed output exists. The source-bound consumed tests were not rewritten.
No additional GPU command was run.

## Next artifact

Create `p4_b0_chunked_prefill_cohort_barrier_design_and_cpu_proof`. It may
specify and test the scheduler barrier, but it must not authorize a GPU probe
or V10. Those remain two separate, source-bound decisions.
