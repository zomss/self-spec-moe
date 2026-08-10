# Phase 97 design review — P1 four-class proxy boots

Status: **historical diagnostic registration, superseded for active design by
the shared-KV-only scope in `README.md`. Draft-KV quantization and every
private/mirrored KV class are closed. The authorization recorded below was
consumed by the completed diagnostics and does not authorize follow-up GPU
runs.**

This document is retained unchanged below this notice for experimental
provenance. Its private-KV classes, gates, and proposed next measurements are
not part of the active Phase 97 work plan.

Authorization: on 2026-08-08 the user requested that the four boot classes
start on GPUs 4–5.

## Objective

Measure whether the four draft-weight/draft-KV combinations boot on the
Phase 96 dense-model stack, how much KV capacity remains after capture, and
whether one deterministic speculative request completes. These runs establish
resource lower bounds before co-resident runtime paths are implemented.

## Important limitation

The current engine constructs one boot-static draft model and one KV
realization. It cannot yet keep a baseline path and an optional quantized path
co-resident and select between them by `action_id`.

Therefore `weight_q`, `kv_q`, and `weight_q_kv_q` below are **static proxies**
for the future capability classes. They measure the selected quantized
realization, not the extra module/graph cost of retaining its baseline fallback.
They must not be reported as runtime-switching results.

## Frozen boot protocol

- Target: `Qwen/Qwen3-8B`, TP1.
- Optional weight-q path:
  `/data/smcho/ckpts/Qwen3-8B-W4A8-gptq` with the Phase 96 Humming kernel
  exclusions.
- Optional KV-q path: private draft KV with `fp8` cache dtype.
- K: 4; no window and no layer skip, to isolate quantization residency.
- `max_model_len=20480`, `max_num_seqs=32`,
  `max_num_batched_tokens=8192`, `gpu_memory_utilization=0.90`.
- Prefix caching off, asynchronous scheduling on, FlashInfer autotuning off.
- The production self-spec CPU orchestration, light metadata, full draft CUDA
  graph, and piecewise-chain flags remain enabled.
- Each boot runs one greedy, fixed 16-token smoke request after capture.
- GPU 4 runs `baseline` then `weight_q`; GPU 5 runs `kv_q` then
  `weight_q_kv_q`.

## Proxy matrix

| ID | draft weights | draft KV | fixed realization | future class represented |
| --- | --- | --- | --- | --- |
| `baseline` | target-matching, aliased | shared target KV | weight-q OFF, KV-q OFF | baseline |
| `weight_q` | W4A8 | shared target KV | weight-q ON, KV-q OFF | baseline + optional weight-q |
| `kv_q` | target-matching, aliased | private fp8 | weight-q OFF, KV-q ON | baseline + optional KV-q |
| `weight_q_kv_q` | W4A8 | private fp8 | weight-q ON, KV-q ON | baseline + both optional paths |

For target-matching classes, `VLLM_SELF_SPEC_SHARE_WEIGHTS=1` is mandatory.
A failure must be reported; silently loading a duplicate target-weight copy is
not an allowed fallback. Shared target KV is explicit for the two KV-q-OFF
classes because the repository environment default remains off.

## Recorded evidence

Each class writes a raw JSON record containing:

- exact environment and engine arguments;
- boot time and smoke-request result;
- `num_gpu_blocks`, block size, engine-reported KV token capacity, and maximum
  concurrency when exposed by vLLM;
- GPU memory before boot, after capture, after smoke, sampled peak, and after
  engine shutdown;
- speculative accepted/draft counters; and
- failure traceback when boot or smoke validation fails.

The aggregate ledger reports capacity relative to `baseline`. GPU memory at
90% utilization is expected to be similar across successful classes because
vLLM fills the residual budget with KV cache; KV token/block capacity is the
primary resource comparison.

## P1 proxy gates

1. **Isolation:** refuse to start a class if its assigned GPU already uses more
   than 2 GiB; never kill an existing process.
2. **Boot:** engine construction and graph capture complete without OOM.
3. **Smoke correctness:** the deterministic request returns 16 tokens and
   speculative draft counters advance.
4. **Capacity:** engine-reported KV capacity is present and exceeds one
   `max_model_len` request.
5. **Cleanup:** GPU memory returns below 2 GiB before the next class starts.

These gates establish feasibility only. Token identity, co-resident path
switching, RL weight refresh, KV readiness/re-entry, workload capacity, and
service value remain later gates in `README.md`.

## Diagnostic amendment after the combined-class stall

The registered `weight_q_kv_q` proxy reached draft AOT compilation but did not
complete its initial draft profiling/warmup run. It remained at about 24 GiB
allocated, 100% SM activity, and 0% device-memory activity for more than
13 minutes, before KV-pool sizing. The owned process was stopped after the
15-minute operational threshold and GPU 5 returned to 0 MiB.

One non-scored localization run is permitted with only
`max_num_batched_tokens` reduced from 8192 to 1024. All other class settings
remain fixed. Success would localize the stall to the large draft-prefill
profiling shape; another stall would implicate the W4A8/private-fp8
realization more generally. Its capacity is diagnostic and is not substituted
into the four-class ledger.

## Registered MBT ladder and scoped eager control

Authorization: on 2026-08-08 the user requested MBT2048 and MBT4096 tests and
isolation of compiled versus eager warmup behavior on GPUs 4–5.

The follow-up is a non-scored 2x2 diagnostic for `weight_q_kv_q`. It changes
only `max_num_batched_tokens` and the draft compilation mode:

| MBT | compiled draft | scoped eager draft |
| ---: | --- | --- |
| 2048 | GPU 4 | GPU 5 |
| 4096 | GPU 5 | GPU 4 |

`compiled` is the frozen default. `eager` sets only
`VLLM_SELF_SPEC_DRAFT_EAGER=1`: the target remains compiled and all other
engine, private-fp8-KV, graph, model, K, memory, and smoke settings remain
fixed. This is deliberately narrower than global `enforce_eager=True`, which
would also change target execution and graph-memory accounting.

Each case uses a separate external compile-cache root to prevent concurrent
cache writes. The two cases at one MBT may run in parallel because this is a
pass/stall localization, not a boot-time comparison. GPU assignment swaps at
MBT4096 to avoid binding one execution mode to one device.

The operational threshold is 900 seconds per case. A timeout is evidence, not
a missing run: preserve the log, identify the last completed warmup stage,
record cleanup, and never kill a process not owned by the runner. A case passes
only if boot, KV sizing, the 16-token speculative smoke, and cleanup complete.

Interpretation is frozen before execution:

- compiled stalls while scoped eager passes at the same MBT: localize the
  boundary to the compiled draft forward or its initial profiling execution;
- both modes stall: implicate the large eager draft shape or another shared
  W4A8/private-KV path rather than compilation alone;
- both modes pass: move the compiled boundary above that MBT; and
- eager stalls while compiled passes: reject eager as a useful isolating
  control and inspect eager-only kernel behavior.

Neither eager capacity nor boot time replaces the registered compiled class.
This diagnostic does not establish serving throughput or runtime switching.

## Registered private-BF16 diagnostic control

Authorization: on 2026-08-08 the user requested a W4A8 plus private-BF16-KV
run while reaffirming that KV-q-OFF means one target-owned KV cache shared by
the target and draft.

That shared-KV rule remains a design invariant. Private BF16 is **not** a
fifth boot class or runtime lever. It is one non-scored control that changes
the failing MBT2048 compiled configuration's draft KV dtype from fp8 to
bfloat16 while keeping private ownership. Its sole purpose is to separate an
fp8-specific failure from a private-KV-path failure.

Frozen control:

- diagnostic id: `diagnostic_weight_q_private_bf16`;
- W4A8 draft weights, private bfloat16 draft KV, target bfloat16 KV;
- compiled draft, MBT2048, K4, `max_model_len=20480`, 32 sequences, and 90%
  GPU-memory utilization;
- all other graph, orchestration, kernel, and 16-token smoke settings match
  the registered combined proxy;
- GPU 4, separate external cold compile cache, and a 900-second threshold.

Interpretation is frozen before execution:

- pass: the failure requires the fp8 private-KV realization;
- same first-draft-forward timeout: private ownership/attention is sufficient
  and fp8 dtype is not required; and
- any earlier failure: inspect the explicit bfloat16 private-cache
  construction before interpreting the warmup hypothesis.

Regardless of outcome, the deployable KV-q-OFF path remains
`VLLM_SELF_SPEC_SHARED_KV=1` with no private draft-KV dtype override.

Observed outcome on 2026-08-08: the control completed the 0.39 s target
profiling warmup and the 21.82 s draft compile, then timed out in the first
MBT2048 draft profiling/warmup forward. It stabilized at 23,633 MiB with 100%
SM activity and 0% device-memory activity, never reached KV sizing, and
released GPU 4 to 0 MiB after the runner stopped its owned process.

This selects the second predeclared interpretation: fp8 KV dtype is not
required for the failure. At this stage, the exact operation in the large
W4A8/private-draft-KV path remained unknown; the registered trace below was
added to resolve it. Private BF16 remains diagnostic-only, does not authorize a
fifth class, and weight-q with KV-q OFF continues to mean W4A8 plus the one
target-owned shared KV cache.

## Registered first-forward operator trace

Authorization: after the private-BF16 result, the user requested the next
diagnostic step on 2026-08-08.

The trace is a non-scored localization run, not a boot class or performance
measurement. It retains W4A8 weights, private bfloat16 draft KV, MBT2048, K4,
the compiled target, GPU 4, and the other control settings. The draft alone is
scoped eager because Python submodule hooks do not execute on the compiled
graph path; prior private-fp8 controls reproduced the same timeout in compiled
and scoped-eager modes.

`VLLM_SELF_SPEC_DRAFT_TRACE=operator` is opt-in and valid only with
`VLLM_SELF_SPEC_DRAFT_EAGER=1`. On the first non-capture draft dummy forward
whose token count equals the configured MBT, it:

1. logs entry and exit for every decoder layer and leaf operator;
2. synchronizes the device after each operator before logging exit, so a
   launched asynchronous kernel cannot be mistaken for a completed operator;
3. removes all hooks if the forward returns; and
4. has no effect when the trace variable is empty, which remains the default.

The operational threshold is 360 seconds. Interpretation is frozen before the
run:

- an unmatched operator entry identifies the first non-returning operation;
- a completed trace followed by a later stall moves localization outside the
  decoder forward;
- a passing boot means per-operator synchronization removes the failure and
  implicates cross-operator asynchrony; and
- no trace-begin marker means the run failed before the intended forward.

Regardless of outcome, production weight-q with KV-q OFF remains W4A8 plus
the one target-owned cache shared with the draft. The private BF16 trace is
diagnostic-only.

Observed outcome on 2026-08-08: the trace began with 432 registered modules.
Layer 0's input RMSNorm logged both entry and synchronized exit. The next
marker was:

```text
enter module=model.layers.0.self_attn.qkv_proj type=QKVParallelLinear
```

No matching exit appeared before the 360-second threshold. The run stabilized
at 23,633 MiB; among 719 telemetry samples, 472 reported 100% SM activity and
713 reported 0% device-memory activity. It did not reach KV sizing and the
owned worker released GPU 4 to 0 MiB.

The unmatched entry identifies the first non-completing work as the layer-0
Humming W4A8 QKV projection or CUDA work launched by it. Rotary embedding and
private-KV attention are downstream and were not reached. The failure must no
longer be described as an unknown operator or a private-KV attention stall.
The private-KV realization remains the interaction context because the W4A8
shared-KV class passes, but this trace does not distinguish a Humming defect
from an input/allocation-layout interaction.

## Registered Cutlass kernel-isolation control

Authorization: on 2026-08-08 the user requested the next debugging step after
reviewing the native vLLM FP8-KV kernel path.

This is a non-scored implementation control, not a fifth boot class. It starts
from the failed private-fp8, MBT2048, scoped-draft-eager case and changes only
the W4A8 linear-kernel selection from Humming to
`CutlassW4A8LinearKernel`. Target execution, private FP8 draft KV, MBT2048,
K4, model length 20480, 32 sequences, 90% GPU-memory utilization,
orchestration settings, and the 16-token smoke request remain fixed.

Frozen control:

- diagnostic id: `diagnostic_weight_q_kv_q_cutlass`;
- GPU 5, matching the failed MBT2048 scoped-eager Humming control;
- separate external cold compile-cache root and a 900-second threshold;
- `kernel_config.linear_backend="cutlass"`, which filters mixed-precision
  candidates to Cutlass and fails model construction if a relevant W4A8
  layer cannot use it; and
- startup must log `Using CutlassW4A8LinearKernel for
  CompressedTensorsW4A8Fp8`. A missing selection line makes the result
  uninterpretable rather than a pass or failure of the hypothesis.

Interpretation is frozen before execution:

- boot, KV sizing, and smoke pass: Humming is required for the observed stall;
  register and run the same control with the draft compiled;
- timeout again inside the first QKV projection: Humming is not required and
  the investigation moves to a kernel-independent W4A8/input-layout
  interaction;
- an explicit Cutlass error: record a Cutlass compatibility failure and do not
  interpret it as either hypothesis outcome; and
- any failure before verified selection: fail closed and repair only the
  forcing/evidence mechanism.

The conditional compiled confirmation changes only draft execution from
scoped eager to compiled. It is authorized only if the eager control passes;
its success criterion is the same boot, KV-sizing, and smoke completion. No
Cutlass result changes the deployable KV-q-OFF invariant: that realization
continues to use one target-owned cache shared by target and draft.

Observed eager outcome on 2026-08-08: startup verified
`CutlassW4A8LinearKernel`, the target profiling warmup completed in 0.38 s,
the MBT2048 draft warmup returned, KV sizing produced 234,320 tokens, graph
capture completed, and the 16-token smoke request passed. Boot took 71.24 s;
the smoke request took 1.70 s, and its counters recorded 12 accepted tokens
and 3 draft events. GPU 5 returned to 0 MiB after shutdown.

This selects the first predeclared interpretation: the observed MBT2048 stall
requires the Humming realization. The passing eager result activates the
registered compiled Cutlass confirmation; it does not yet promote the
combined boot class.

Observed compiled outcome on 2026-08-08: startup again verified
`CutlassW4A8LinearKernel`. Target and draft compilation completed in 16.55 s
and 31.61 s total, their profiling warmups returned in 0.40 s and 0.42 s, KV
sizing produced 232,608 tokens, graph capture completed, and the 16-token smoke
request passed. Boot took 94.29 s, smoke took 0.76 s, and GPU 5 returned to
0 MiB after shutdown.

The confirmation shows that disabling draft compilation is not needed once
Cutlass replaces Humming. The combined Humming realization remains blocked;
the Cutlass realization is now eligible for a separately registered MBT8192
production-shape boot control, not yet for runtime profiling.

## Commands and artifacts

```bash
bash research/97_composition_runtime/scripts/run_boot_matrix.sh
bash research/97_composition_runtime/scripts/run_warmup_diagnostics.sh
bash research/97_composition_runtime/scripts/run_private_bf16_diagnostic.sh
bash research/97_composition_runtime/scripts/run_private_bf16_trace.sh
bash research/97_composition_runtime/scripts/run_cutlass_kernel_control.sh
```

Raw records and logs:

```text
data/boot_proxy/<class>.json
data/boot_resource_ledger.json
data/diagnostics/weight_q_kv_q_mbt1024.json
data/diagnostics/weight_q_kv_q_mbt{2048,4096}_{compiled,eager}.json
data/diagnostics/weight_q_private_bf16_mbt2048_compiled.json
data/warmup_diagnostic_ledger.json
logs/boot_proxy/<class>.log
logs/diagnostics/weight_q_kv_q_mbt1024.log
logs/diagnostics/weight_q_kv_q_mbt{2048,4096}_{compiled,eager}.log
logs/diagnostics/weight_q_private_bf16_mbt2048_compiled.log
logs/diagnostics/weight_q_private_bf16_mbt2048_compiled.gpu.csv
data/diagnostics/weight_q_private_bf16_mbt2048_operator_trace_eager.json
logs/diagnostics/weight_q_private_bf16_mbt2048_operator_trace_eager.log
logs/diagnostics/weight_q_private_bf16_mbt2048_operator_trace_eager.gpu.csv
data/diagnostics/weight_q_kv_q_mbt2048_eager_cutlass.json
logs/diagnostics/weight_q_kv_q_mbt2048_eager_cutlass.log
logs/diagnostics/weight_q_kv_q_mbt2048_eager_cutlass.gpu.csv
data/diagnostics/weight_q_kv_q_mbt2048_compiled_cutlass.json
logs/diagnostics/weight_q_kv_q_mbt2048_compiled_cutlass.log
logs/diagnostics/weight_q_kv_q_mbt2048_compiled_cutlass.gpu.csv
```

Observed follow-up: MBT2048 and MBT4096 time out in both compiled and scoped
draft-eager modes, and the private-BF16 MBT2048 control reproduces the same
first-draft-forward timeout. The operator trace localizes the first unfinished
work to layer 0's Humming QKV projection, before rotary embedding or
private-KV attention. See `results_warmup_diagnostics.md`. FP8 KV dtype and
draft compilation are not required for the failure. Both forced-Cutlass
MBT2048 controls pass, so the remaining gate is a compiled Cutlass run at the
registered MBT8192 production shape.
