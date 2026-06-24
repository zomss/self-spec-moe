# Phase 02: Timing Envelope Experiment

Date: 2026-06-23

## Goal

Before implementing local expert routing, measure whether the communication
envelope has enough room to justify the next phase.

This phase does **not** measure model quality or real acceptance. It measures
the timing terms needed to answer:

```text
At each batch size and draft length, what acceptance rate beta is required for
local collective-free draft to add speedup on top of DeepEP and DBO-style
reference stacks?
```

Source phase: [`../01_communication_advantage`](../01_communication_advantage)

## Assumptions

- DeepEP and DBO are orthogonal stack components, not competitors.
- Local draft quality is unknown; this phase only measures the timing envelope.
- `t_draft_local_ms` may be estimated in the first pass, but must be labeled as
  estimated until a real no-all-to-all draft path exists.
- Mixed prefill/decode batches are out of scope unless reported separately.

## Target Setting

### Primary Setting

| Dimension | Target |
| --- | --- |
| Deployment | Multi-node expert-parallel MoE decode |
| Serving mode | Decode-only path; prefill separated or reported separately |
| Communication backend | DeepEP low-latency for exact verification collectives |
| Overlap stack | DBO enabled when supported, plus phase-overlap variant if prototyped |
| Batch range | Low-to-moderate decode batch: `B = 1, 2, 4, 8, 16` |
| Draft lengths | `k = 1, 2, 4, 8` |
| Output metric | TPOT / per-cycle latency envelope, not end-to-end serving quality |

### Secondary Sanity Setting

Single-node EP can be used only as a sanity/debug setting. It is useful for
checking scripts and measurement methodology, but it is not expected to show a
strong systems result because NVLink/NVSwitch all-to-all is comparatively fast.

## Reference Stacks

Treat DeepEP and DBO as orthogonal components, not competitors.

| `reference_stack` | Meaning |
| --- | --- |
| `plain_ep_verify` | Exact autoregressive EP decode without DeepEP/DBO-specific optimization, if available. |
| `deepep_only_verify` | Exact autoregressive decode using DeepEP low-latency. |
| `deepep_dbo_verify` | Exact autoregressive decode using DeepEP low-latency plus DBO. |
| `deepep_phase_overlap_verify` | Optional: exact verification with phase-offset overlap if prototyped. |

Self-MoE-spec should be evaluated as incremental speedup on top of these
reference stacks.

## Measurements

The calculator expects one row per `(reference_stack, B, k)`:

| Column | Meaning | How to obtain |
| --- | --- | --- |
| `reference_stack` | Stack being improved on | One of the names above, or a more specific run name |
| `batch_size` | Decode batch size `B` | Scheduler/request setup |
| `draft_length` | Draft tokens per cycle `k` | Sweep value |
| `t_base_ms` | One-token decode latency of the reference stack | Measured from exact autoregressive decode |
| `t_draft_local_ms` | One local draft step with no expert all-to-all | Measured by prototype or estimated from no-comm components |
| `t_verify_ms` | Exact verification latency for `B * (k + 1)` tokens | Measured by batched exact verification shape |

### Important Measurement Rules

- Report p50, p90, and p99 separately if possible; use p50 for envelope
  discovery and p90/p99 for SLO discussion.
- Warm up CUDA graphs/kernels before recording.
- Keep model, parallelism, prompt length, output length, sampling settings, and
  hardware fixed within a sweep.
- Record whether prefill is absent, disaggregated, or mixed. Mixed prefill
  invalidates the draft-collective-free assumption unless reported separately.
- Include scheduler/cycle-boundary overhead if it exists in the measured path.
- For `t_verify_ms`, use the actual verification shape `B * (k + 1)`, not just
  the one-token decode latency.

## Local Draft Timing Options

`t_draft_local_ms` is the hardest term before implementation. Use the most
realistic available option and label it in notes/logs:

1. **Prototype measurement:** real local-only MoE draft path with all-to-all
   disabled. This is best, but may require implementation.
2. **Component estimate:** `t_base_ms - exposed_all_to_all_ms`, measured from
   profiling exact decode.
3. **Optimistic bound:** `t_base_ms * (1 - f_exposed)` from profiler-estimated
   exposed communication fraction.
4. **Kernel microbench:** local expert grouped GEMM plus attention path without
   communication, if available.

The first pass may use options 2-3. The result must be labeled as an envelope,
not a measured implementation speedup.

## Commands

Use the Phase 01 calculator with Phase 02 timing inputs:

```bash
.venv/bin/python research/01_communication_advantage/speedup_envelope.py \
    research/02_timing_envelope_experiment/data/timings_<system>.csv \
    --target-speedup 1.3 --format csv \
    > research/02_timing_envelope_experiment/data/envelope_<system>.csv
```

For acceptance-rate sweeps:

```bash
.venv/bin/python research/01_communication_advantage/speedup_envelope.py \
    research/02_timing_envelope_experiment/data/timings_<system>.csv \
    --acceptance-rates 0.5,0.6,0.7,0.8,0.9,1.0 --format csv \
    > research/02_timing_envelope_experiment/data/acceptance_sweep_<system>.csv
```

## Target Outputs

All outputs should stay in `research/02_timing_envelope_experiment/`.

| Output | Path | Purpose |
| --- | --- | --- |
| Timing CSV | `data/timings_<system>.csv` | Raw timing inputs for `speedup_envelope.py` |
| Envelope CSV | `data/envelope_<system>.csv` | `S_max`, `beta_min`, and verdicts |
| Acceptance sweep CSV | `data/acceptance_sweep_<system>.csv` | Speedup curves over beta |
| Summary table | `results_<system>.md` | Human-readable interpretation and go/no-go |
| Figures | `figures/*.png` or `figures/*.pdf` | Plots of `S_max(B,k)` and `beta_min(B,k)` |
| Logs | `logs/*.txt` | Commands, environment, and raw benchmark snippets |

## Local Sweep Helper

`run_latency_sweep.py` loads one vLLM engine and sweeps multiple batch sizes.
Use it for sanity runs or for collecting the `t_base_ms` and `t_verify_ms`
terms when the target model/backend is available.

`run_dp_latency_sweep.py` is the multi-process variant for local DP+EP sweeps.
It is intended for Phase 02 reference-stack timing when the target model and
backend can run on the current node.

`emulate_multinode_from_timings.py` uses measured verification-shape scaling and
synthetic exposed-communication fractions to emulate multi-node envelope
scenarios before real multi-node DeepEP measurements are available.

`emulate_comm_delay.py` directly injects synthetic exposed communication delay
into the timing model. Use it to answer how much multi-node-like communication
is needed before Self-MoE-spec becomes worthwhile.

The codebase now also has a research-only runtime hook:
`VLLM_SELF_SPEC_EMULATE_A2A_DELAY_MS`. It injects host-side delay after the MoE
EP allgather/reducescatter path when using `allgather_reducescatter`. The delay
is per MoE collective call, not per decode step.

## Current Result

`results_sanity_tp2_ep_est_f15.md` records a completed single-node TP2+EP dummy
MoE sanity run. It validates the workflow but is not the primary multi-node
DeepEP/DBO result.

`results_dp2_gpus6_7.md` records the requested GPU-6/7 DP2+EP probe. It
successfully measures the allgather reference stack and documents that
DeepEP/DBO are blocked because DeepEP kernels are not installed.

`results_emulated_multinode_from_dp2.md` uses the GPU-6/7 shape-scaling data to
emulate multi-node exposed-communication regimes (`f=0.2,0.4,0.6`) and DBO
composition sensitivity.

`results_comm_delay_emulation.md` directly injects exposed communication delay
and shows that useful speedup begins when communication delay is roughly
comparable to local compute time.

`results_forced_ib_gpus6_7.md` documents that GPUs 6 and 7 can be forced onto
IB/GDRDMA transport with NCCL environment variables, but the vLLM probe hung
before producing a timing result.

`results_forced_socket_gpus6_7.md` resolves the hanging issue by forcing NCCL to
use the socket network path instead of IB/GDRDMA. This is the stable local
network-emulation mode for GPUs 6 and 7.

`code_opinion_forced_network.md` traces the vLLM all-to-all code path and
recommends delay injection in the MoE EP all-to-all manager rather than trying
to force IB/GDRDMA inside vLLM.

`current_conclusion_and_next_step.md` summarizes the current Phase 02 conclusion
and the next calibrated delay-sweep plan.

`results_calibrated_delay_sweep.md` reports the GPU6/7 delay-hook sweep for
`VLLM_SELF_SPEC_EMULATE_A2A_DELAY_MS=0,0.025,0.05,0.1,0.2`.

`practical_delay_scale.md` explains how to map IB-scale latency and total
per-token communication budgets to the per-collective delay hook.

`results_a2a_count_calibration.md` measures how often the current hook fires and
derives the smaller delay values needed for the next calibrated runtime sweep.

## Required Tables

### Table 1: Timing Inputs

| reference_stack | B | k | `t_base_ms` | `t_draft_local_ms` | `t_verify_ms` |
| --- | ---: | ---: | ---: | ---: | ---: |
| deepep_only_verify | 1 | 4 | TBD | TBD | TBD |
| deepep_dbo_verify | 1 | 4 | TBD | TBD | TBD |

### Table 2: Envelope

| reference_stack | B | k | `S_max` | `beta_min@1.0x` | `beta_min@1.3x` | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| deepep_only_verify | 1 | 4 | TBD | TBD | TBD | TBD |
| deepep_dbo_verify | 1 | 4 | TBD | TBD | TBD | TBD |

### Table 3: Acceptance Sweep

| reference_stack | B | k | beta=0.7 | beta=0.8 | beta=0.9 | beta=1.0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| deepep_only_verify | 1 | 4 | TBD | TBD | TBD | TBD |
| deepep_dbo_verify | 1 | 4 | TBD | TBD | TBD | TBD |

## Required Figures

1. `S_max` heatmap over `(B, k)` for each reference stack.
2. `beta_min@1.0x` heatmap over `(B, k)`.
3. `beta_min@1.3x` heatmap over `(B, k)`.
4. Speedup-vs-acceptance curves for selected `B` and `k`.

## Go/No-Go Criteria

Proceed to Phase 03 local-routing acceptance measurement if the measured timing
envelope satisfies at least one:

- `beta_min@1.3x <= 0.9` on top of `deepep_only_verify` for a target
  multi-node low-batch setting.
- `S_max >= 1.15` on top of `deepep_dbo_verify` or phase-overlap for a target
  low-batch setting.
- The envelope shows a clear path to improving DBO via cheaper draft or
  phase-offset scheduling.

Do not proceed directly to runtime implementation if:

- `S_max <= 1.1` for the optimized reference stack,
- `beta_min@1.0x > 0.9` for most useful `(B, k)` settings,
- gains only appear when DeepEP/DBO are disabled or misconfigured.

## Expected Next Artifact

The next artifact should be a measured timing input:

```text
research/02_timing_envelope_experiment/data/timings_<system>.csv
```

followed by generated envelope outputs from the Phase 01
`speedup_envelope.py`.
