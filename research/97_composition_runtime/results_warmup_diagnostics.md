# Phase 97 P1 result — combined-class warmup localization

Status: **complete kernel localization. Humming times out from MBT2048 in both
compiled and scoped draft-eager paths, while forced Cutlass passes MBT2048 in
both modes. The blocker requires the Humming W4A8 realization; it is not an
FP8-KV attention or draft-compilation failure.**

Source: the registered amendment in `design_review.md`, run on GPUs 4–5 on
2026-08-08. These are non-scored static boot diagnostics, not serving or
runtime-switching measurements.

## Question

The combined W4A8-weight/private-fp8-KV proxy passed at MBT1024 but stalled at
MBT8192 after draft AOT compilation. The diagnostic asks:

1. does the first registered failure occur at MBT2048 or MBT4096;
2. does disabling compilation only for the draft make the profiling forward
   return; and
3. does retaining private ownership but changing the draft KV dtype from fp8
   to bfloat16 make the MBT2048 profiling forward return; and
4. if not, which operation first fails to complete in that forward?
5. does forcing every relevant W4A8 linear to Cutlass make the same MBT2048
   eager and compiled realizations complete?

The scoped eager mode sets `VLLM_SELF_SPEC_DRAFT_EAGER=1`. The target remains
compiled, so this isolates draft compilation without changing the target or
globally disabling CUDA graphs. The BF16 control is diagnostic-only: it keeps
private draft-KV ownership and changes only its dtype. It does not redefine
KV-q-OFF, whose deployable realization remains one target-owned KV cache shared
by target and draft.

## Result

| MBT | draft mode | GPU | target initial warmup | draft compile | draft initial warmup | terminal result |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1024 | compiled | 5 | 0.35 s | 31.42 s total | 0.41 s | pass; 234,528 KV tokens and smoke pass |
| 2048 | compiled | 4 | 0.37 s | 22.56 s total | did not return | timeout at 900 s |
| 2048 | scoped eager | 5 | 0.36 s | disabled | did not return | timeout at 900 s |
| 4096 | compiled | 5 | 0.43 s | 23.23 s total | did not return | timeout at 900 s |
| 4096 | scoped eager | 4 | 0.44 s | disabled | did not return | timeout at 900 s |
| 8192 | compiled | 5 | 0.59 s | 25.33 s total | did not return | stopped after the 900 s threshold |

The registered kernel-isolation controls changed only the W4A8 backend:

| MBT | W4A8 kernel | draft mode | selection | target warmup | draft compile/warmup | KV tokens | smoke |
| ---: | --- | --- | --- | ---: | --- | ---: | --- |
| 2048 | Cutlass | scoped eager | verified | 0.38 s | disabled; forward returned | 234,320 | 16/16 pass |
| 2048 | Cutlass | compiled | verified | 0.40 s | 31.61 s / 0.42 s | 232,608 | 16/16 pass |

Both used `kernel_config.linear_backend="cutlass"`, and both logs contain
`Using CutlassW4A8LinearKernel for CompressedTensorsW4A8Fp8`. Explicit backend
filtering fails model construction if a relevant layer cannot use Cutlass, so
selection evidence is fail closed. Eager and compiled boot took 71.24 s and
94.29 s respectively, but these cold diagnostic boot times are not compared.

The rows above use private fp8 draft KV. The additional private-BF16 control
at MBT2048 on GPU 4 completed target warmup in 0.39 s and draft compilation in
21.82 s, but its first draft profiling/warmup forward also failed to return and
timed out at 900 s.

The follow-up private-BF16 operator trace kept MBT2048, K4, the compiled
target, and the Humming W4A8 realization, but used scoped draft eager so Python
hooks could execute. It reached this sequence:

```text
begin operator trace (432 modules)
enter model.layers.0
enter model.layers.0.input_layernorm
exit  model.layers.0.input_layernorm
enter model.layers.0.self_attn.qkv_proj (QKVParallelLinear)
```

No matching QKV-projection exit appeared before the 360-second threshold.
Every post-hook synchronizes the CUDA device before logging its exit, so the
unmatched entry localizes the first unfinished work to that projection or CUDA
work launched by it; no downstream operator started.

The 2048 runs stabilized at 23,665 MiB and the 4096 runs at 23,789 MiB. All
four showed 100% SM activity, 0% device-memory activity, and no log progress
until timeout. None reached KV-pool sizing, graph capture, or the smoke request.
The owned workers terminated cleanly and both GPUs returned to 0 MiB after
each pair.

The BF16 control stabilized at 23,633 MiB. Its 1,798 half-second telemetry
samples included 1,511 at 100% SM activity and 1,794 at 0% device-memory
activity. It never reached KV-pool sizing, graph capture, or the smoke request,
and GPU 4 returned to 0 MiB after the owned worker was stopped.

The operator trace also stabilized at 23,633 MiB. Its 719 samples included 472
at 100% SM activity and 713 at 0% device-memory activity. It likewise never
reached KV sizing, graph capture, or smoke, and GPU 4 returned to 0 MiB.

Boot times are not compared across modes: each case used a separate cold
compile-cache root and paired runs shared host resources. Only completion and
the last reached stage are interpreted.

## Isolation

The compile hypothesis is rejected for this failure:

- both compiled cases finished draft AOT generation before the stall;
- the scoped eager cases never compiled the draft and stalled in the same
  forward with the same GPU signature; and
- target compilation and target initial warmup completed in every case.

The source path narrows the blocked operation to the first large dummy draft
forward. `GPUModelRunner.profile_run()` drives `_dummy_run()` with
`max_num_tokens`; that calls `drafter.dummy_run()` with the MBT-sized draft
batch, which enters the draft model forward. The first draft forward does not
return at MBT2048 or above in this realization.

The existing class controls further narrow the interaction:

- W4A8 draft weights with shared target KV pass at MBT8192;
- target-matching draft weights with private fp8 KV pass at MBT8192; and
- W4A8 plus private fp8 KV passes at MBT1024; but
- W4A8 plus private bfloat16 KV reproduces the MBT2048 timeout after both
  target and draft compilation complete.

Therefore neither optional representation alone is sufficient to produce the
failure. It is specific to the large W4A8/private-draft-KV profiling path, and
changing the private cache from fp8 to bfloat16 does not repair it.

The operator trace closes the layer/operator ambiguity. Layer 0's
input RMSNorm completes, then its `QKVParallelLinear` projection is the first
operation whose synchronized exit never appears. Rotary embedding, attention,
and any private-KV read/write are downstream and are not reached. The result
therefore does **not** implicate the private-KV attention kernel itself. It
localizes the failure to the first Humming W4A8 QKV projection under the
private-KV realization/memory layout.

The forced-backend contrast closes the kernel ambiguity at MBT2048. With all
other registered inputs fixed, Cutlass completes that forward, KV sizing,
graph capture, and smoke in both draft execution modes. Therefore Humming is
required for the observed stall. This does not prove an isolated Humming
implementation defect: the causal scope remains the interaction between
Humming and this large private-draft-KV realization. It does prove that an FP8
KV or private-attention workaround targets the wrong component.

## Decision

- Keep the **Humming** combined realization blocked from MBT2048 upward.
- Treat Cutlass as the valid repair candidate: its eager and compiled MBT2048
  controls both boot and smoke successfully.
- Do not promote the Cutlass combined class at MBT8192 yet. MBT2048 is a
  kernel-isolation control, not the registered production-shape capacity run.
- Do not pursue global eager mode as the fix; the narrower eager control has
  already removed draft compilation without changing the outcome.
- Do not promote private BF16 to a boot class or use it as a KV-q-OFF
  fallback. The deployable W4A8/KV-q-OFF path remains the passing shared-target
  KV realization.
- MBT1024 remains a diagnostic proof of HBM fit, not a substitute for the
  production boot class.
- Do not merely cap boot profiling at 1024. Runtime private-KV prefill can
  encounter larger draft batches, and a capped profile would also understate
  activation memory unless a conservative reserve is added.
- Do not design a private-KV attention workaround for this symptom; that
  operator is not reached.

The next controlled experiment should use the compiled Cutlass realization at
the registered production shape, MBT8192. It must retain fail-closed backend
selection evidence, use a fresh cold cache root, and remeasure boot completion,
KV capacity, graph capture, and smoke before the combined class enters runtime
profiling. If MBT8192 fails, MBT4096 becomes the threshold-localization control.

## Artifacts

- `data/warmup_diagnostic_ledger.json`: registered 2x2 matrix, private-BF16
  control, operator trace, telemetry counts, and parsed log milestones.
- `data/diagnostics/weight_q_kv_q_mbt{2048,4096}_{compiled,eager}.json`:
  terminal records.
- `logs/diagnostics/weight_q_kv_q_mbt{2048,4096}_{compiled,eager}.log`: full
  engine evidence.
- `data/diagnostics/weight_q_private_bf16_mbt2048_compiled.json`: terminal
  record for the private-BF16 control.
- `logs/diagnostics/weight_q_private_bf16_mbt2048_compiled.{log,gpu.csv}`:
  full engine and GPU telemetry evidence for the private-BF16 control.
- `data/diagnostics/weight_q_private_bf16_mbt2048_operator_trace_eager.json`:
  terminal trace record and unmatched operator marker.
- `logs/diagnostics/weight_q_private_bf16_mbt2048_operator_trace_eager.{log,gpu.csv}`:
  full trace and GPU telemetry evidence.
- `scripts/run_warmup_diagnostics.sh`: GPU ownership, timeout, cleanup, and
  matrix runner.
- `scripts/run_private_bf16_diagnostic.sh`: isolated BF16-control runner.
- `scripts/run_private_bf16_trace.sh`: isolated, scoped-eager operator trace.
- `data/diagnostics/weight_q_kv_q_mbt2048_{eager,compiled}_cutlass.json`:
  passing forced-Cutlass records.
- `logs/diagnostics/weight_q_kv_q_mbt2048_{eager,compiled}_cutlass.{log,gpu.csv}`:
  backend-selection, engine-stage, smoke, and GPU evidence.
- `scripts/run_cutlass_kernel_control.sh`: fail-closed eager/compiled Cutlass
  runner.

The diagnostic ledger has `matrix_complete=true`,
`additional_control_complete=true`, `operator_trace_complete=true`,
`kernel_controls_complete=true`, `kernel_controls_all_booted=true`, and
`all_booted=false` for the original Humming matrix.
