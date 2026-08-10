# Phase 97 P1 result — four-class static boot proxies

Status: **three primary proxy boots passed; the original combined
production-shape proxy is blocked in its first Humming W4A8 QKV projection.
Forced Cutlass repairs the same combined realization at MBT2048 in both eager
and compiled draft modes; production-shape MBT8192 remains untested.**

Source: `design_review.md`, user-authorized GPUs 4–5 run on 2026-08-08.
These are non-scored resource measurements, not runtime-switching or serving
performance results.

## Protocol

All primary runs use Qwen3-8B TP1, K4, no window/skip,
`max_model_len=20480`, `max_num_seqs=32`,
`max_num_batched_tokens=8192`, 90% GPU memory utilization, fixed graph
settings, FlashInfer autotuning off, and a greedy 16-token smoke request.

GPU 4 ran `baseline` then `weight_q`; GPU 5 ran `kv_q` then
`weight_q_kv_q`. The runner refused occupied GPUs and never killed an
unrelated process. Both GPUs returned to 0 MiB after the work.

## Primary result

| static proxy | result | boot s | KV tokens | blocks | max 20,480-token concurrency | capacity vs baseline |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| baseline: target-matching weights + shared KV | pass | 65.4 | 392,464 | 24,529 | 19.16x | 100.0% |
| weight-q: W4A8 + shared KV | pass | 149.2 | 350,848 | 21,928 | 17.13x | 89.4% |
| KV-q: target-matching weights + private fp8 KV | pass | 84.2 | 261,472 | 16,342 | 12.77x | 66.6% |
| weight-q + KV-q: W4A8 + private fp8 KV | **stall** | >900 | not reached | not reached | not reached | not reached |

Each passing class returned 16/16 smoke tokens with 12 accepted draft tokens,
3 draft events, and zero preemptions. This is a path-engagement smoke check,
not a general correctness or acceptance result.

The primary combined run loaded the target and W4A8 drafter, prewarmed all
144 Humming linears over 18 M tiles, compiled the target and draft AOT graphs,
and then stopped making progress in the initial draft profiling/warmup run.
For more than 13 minutes it stayed at about 24,011 MiB, 100% SM activity,
0% device-memory activity, and about 120 W. This is before KV-pool sizing, so
the failure is not HBM exhaustion. The owned run was stopped after the
15-minute operational threshold and GPU 5 returned to 0 MiB.

## Reduced-shape localization

One diagnostic changed only `max_num_batched_tokens` from 8192 to 1024:

| static proxy | result | boot s | KV tokens | blocks | max concurrency | capacity vs primary baseline |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| weight-q + KV-q, MBT1024 | pass | 156.8 | 234,528 | 14,658 | 11.45x | 59.8% |

It completed its initial profiling runs in 0.35/0.41 seconds, allocated the KV
pool, captured graphs in 8 seconds, passed the 16-token smoke, and shut down
cleanly. Together with the later private-BF16 timeout, the production-shape
stall is specific to the large W4A8/private-draft-KV profiling path, not to fp8
dtype or the static representation's HBM fit. The subsequent synchronized
operator trace localizes its first unfinished work to layer 0's Humming QKV
projection, before rotary embedding or private-KV attention.

The diagnostic capacity is also internally consistent. Applying private fp8
draft KV to the weight-q shared-KV capacity predicts roughly
`350,848 / 1.5 = 233,899` tokens; the measured 234,528 differs by only 0.27%.
The 1.5 factor is target bf16 KV plus an additional half-sized fp8 draft KV.

## Kernel isolation

The fail-closed Cutlass control held the failed MBT2048 combined case fixed:

| W4A8 kernel | draft mode | result | boot s | KV tokens | max concurrency | smoke |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Humming | scoped eager | timeout | >900 | not reached | not reached | not reached |
| Cutlass | scoped eager | pass | 71.24 | 234,320 | 11.44x | 16/16 |
| Cutlass | compiled | pass | 94.29 | 232,608 | 11.36x | 16/16 |

Both passing logs verify `CutlassW4A8LinearKernel`; the compiled case records
target/draft warmups of 0.40/0.42 seconds. This establishes that the MBT2048
stall requires Humming. It does not yet establish that Cutlass passes the
primary MBT8192 resource shape or has acceptable steady-state throughput.

## Interpretation

1. **Weight-q has a fixed-memory cost.** The W4A8 draft reduces available KV
   capacity by 10.6% even before reserving a co-resident target-matching graph
   path or RL refresh workspace.
2. **Private KV-q is the larger capacity cost.** It reduces capacity to 66.6%
   of shared target KV, exactly the expected target-bf16-plus-draft-fp8 shape.
3. **Both static representations physically fit.** The diagnostic leaves
   234,528 KV tokens, about 59.8% of baseline, but MBT1024 is not substituted
   for the failed primary configuration.
4. **The combined Humming realization is not deployable from MBT2048.**
   Cutlass repairs MBT2048, but the primary MBT8192 shape must still be booted
   and measured before throughput or selector evaluation.
5. **These remain lower bounds for future capability classes.** The current
   engine cannot co-reside the baseline and optional quantized paths. Extra
   modules, graphs, versioning state, and refresh workspace will reduce
   capacity further and must be measured after implementation.

## Decision

- Admit `baseline`, `weight_q`, and `kv_q` to the next co-residency design
  check, subject to workload capacity gates.
- Do not admit `weight_q_kv_q` at MBT8192 to runtime profiling.
- The completed MBT2048/4096 follow-up places the first failing registered
  shape between 1024 and 2048. Scoped draft-eager controls also time out, so
  draft compilation is not the cause; see `results_warmup_diagnostics.md`.
- The W4A8/private-BF16-KV MBT2048 control also times out in the first draft
  profiling forward, so fp8 dtype is not required. The operator trace places
  the failure in the first Humming QKV projection; private-KV attention is not
  reached.
- Keep W4A8 with KV-q OFF on its passing shared-target-KV realization; private
  BF16 is diagnostic-only and is not a fifth boot class.
- The Cutlass-only MBT2048 eager and compiled controls both pass. Do not design
  a profiling, prefill, or attention workaround for the Humming symptom.
- Run the compiled Cutlass combined realization at MBT8192 next. Admit it to
  runtime profiling only if boot, measured capacity, graphs, and smoke pass.
- Do not interpret boot time as steady-state cost; W4A8 includes about
  75–80 seconds of one-time Humming kernel prewarm.

## Artifacts

- `data/boot_resource_ledger.json`: aggregate primary ledger plus diagnostic.
- `data/boot_proxy/*.json`: primary per-class records.
- `data/diagnostics/weight_q_kv_q_mbt1024.json`: reduced-shape diagnostic.
- `results_warmup_diagnostics.md` and `data/warmup_diagnostic_ledger.json`:
  MBT2048/4096 compiled-versus-scoped-eager localization plus the private-BF16
  MBT2048 control, first-forward trace, and Cutlass kernel controls.
- `logs/boot_proxy/*.log` and `logs/diagnostics/*.log`: full engine evidence.

The primary ledger remains `complete=false` by design because the registered
combined production-shape class did not reach boot completion.
