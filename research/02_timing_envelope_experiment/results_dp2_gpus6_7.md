# Results: DP2 EP Probe on GPUs 6 and 7

Date: 2026-06-23

## Status

This run used the requested GPUs 6 and 7. It successfully measured a local
DP2+EP reference stack with `allgather_reducescatter`, but could not measure
DeepEP or DBO because DeepEP kernels are not installed in this environment.

This is still a **sanity result**, not the primary multi-node DeepEP+DBO result.

## Environment

| Item | Value |
| --- | --- |
| GPUs | `CUDA_VISIBLE_DEVICES=6,7` |
| Model config | `Qwen/Qwen1.5-MoE-A2.7B` |
| Weights | `--load-format dummy` |
| Parallelism | `--data-parallel-size 2 --enable-expert-parallel` |
| Successful backend | `allgather_reducescatter` |
| Failed backend | `deepep_low_latency` |
| DBO | Blocked because DBO requires DeepEP |
| Decode shape | `input_len=1`, `output_len=1` |
| Timing statistic | p50 latency |

## Commands

Successful reference-stack probe:

```bash
CUDA_VISIBLE_DEVICES=6,7 .venv/bin/python \
  research/02_timing_envelope_experiment/run_dp_latency_sweep.py \
  --model Qwen/Qwen1.5-MoE-A2.7B \
  --load-format dummy \
  --trust-remote-code \
  --data-parallel-size 2 \
  --enable-expert-parallel \
  --all2all-backend allgather_reducescatter \
  --enforce-eager \
  --max-model-len 128 \
  --input-len 1 \
  --output-len 1 \
  --batch-sizes 2,4,6,10 \
  --num-iters-warmup 1 \
  --num-iters 3 \
  --output-json data/dp2_ep_allgather_gpus6_7.json
```

Attempted DeepEP/DBO probes failed with:

```text
AssertionError: DeepEP kernels not found.
```

## Generated Artifacts

| Artifact | Purpose |
| --- | --- |
| `data/dp2_ep_allgather_gpus6_7.json` | Raw DP2 allgather latency output |
| `data/timings_dp2_ep_allgather_gpus6_7_est_f15.csv` | Envelope timing input |
| `data/envelope_dp2_ep_allgather_gpus6_7_est_f15_break_even.csv` | Break-even envelope |
| `data/envelope_dp2_ep_allgather_gpus6_7_est_f15_target_1p3.csv` | 1.3x target envelope |
| `logs/dp2_ep_allgather_gpus6_7.log` | Successful DP2 allgather log |
| `logs/dp2_ep_deepep_ll_gpus6_7.log` | Failed DeepEP log |
| `logs/dp2_ep_deepep_dbo_gpus6_7.log` | Failed DeepEP+DBO log |

## Measured p50 Latencies

| Global batch size | p50 latency (ms) |
| ---: | ---: |
| 2 | 1536.258 |
| 4 | 1587.600 |
| 6 | 1544.828 |
| 10 | 1529.210 |

## Timing Envelope Input

`t_draft_local_ms` is estimated with `f_exposed = 0.15`, so this is still an
envelope rather than a measured local-draft implementation.

| reference_stack | B | k | `t_base_ms` | `t_draft_local_ms` | `t_verify_ms` |
| --- | ---: | ---: | ---: | ---: | ---: |
| dp2_ep_allgather_gpus6_7_est_f15 | 2 | 1 | 1536.258 | 1305.819 | 1587.600 |
| dp2_ep_allgather_gpus6_7_est_f15 | 2 | 2 | 1536.258 | 1305.819 | 1544.828 |
| dp2_ep_allgather_gpus6_7_est_f15 | 2 | 4 | 1536.258 | 1305.819 | 1529.210 |

## Envelope Results

| B | k | `S_max` | `beta_min@1.0x` | `beta_min@1.3x` |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 1 | 1.062 | 0.883 | NA |
| 2 | 2 | 1.109 | 0.898 | NA |
| 2 | 4 | 1.138 | 0.936 | NA |

## Interpretation

The DP2 allgather result is consistent with the earlier single-node conclusion:

- The envelope can barely break even only at high acceptance.
- It cannot reach 1.3x even under the optimistic `f_exposed=0.15` draft estimate.
- This is not surprising because it is still a single-node, non-DeepEP,
  non-DBO sanity environment.

The primary Phase 02 result still requires installing DeepEP kernels and running
the same workflow for:

1. `deepep_only_verify`,
2. `deepep_dbo_verify`,
3. optionally `deepep_phase_overlap_verify`.

## Next Step

Install or enable DeepEP kernels in this vLLM environment, then rerun the
GPU-6/7 DP2 workflow with `--all2all-backend deepep_low_latency` and DBO.
