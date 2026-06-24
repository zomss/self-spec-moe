# Results: Single-Node TP2 EP Sanity Envelope

Date: 2026-06-23

## Status

This is a **sanity experiment**, not the primary Phase 02 result.

It validates the Phase 02 timing/envelope workflow on the available local
hardware, but it does not answer the main multi-node DeepEP+DBO question.

## Environment

| Item | Value |
| --- | --- |
| Hardware | Single node, 8x H100 80GB available |
| GPUs used | `CUDA_VISIBLE_DEVICES=0,1` |
| Model config | `Qwen/Qwen1.5-MoE-A2.7B` |
| Weights | `--load-format dummy` |
| Parallelism | `--tensor-parallel-size 2 --enable-expert-parallel` |
| All-to-all backend | `allgather_reducescatter` |
| DBO | Not used; offline `LLM` path rejects DP>1 and DBO requires DP+EP |
| DeepEP | Not used in this sanity run |
| Decode shape | `input_len=16`, `output_len=1` |
| Timing statistic | p50 latency |

## Important Caveats

- This is **single-node TP+EP**, not multi-node DP+EP.
- This uses dummy weights and is only a systems-path sanity check.
- `t_draft_local_ms` is **estimated**, not measured. It assumes
  `f_exposed = 0.15`, so `t_draft_local = 0.85 * t_base`.
- `t_verify_ms` is approximated by running separate one-token requests with
  batch size `B * (k + 1)`. This approximates token count, but it is not a real
  speculative verification kernel.
- The offline latency CLI refused `data_parallel_size=2`, so this environment
  cannot directly measure the DBO reference stack through `vllm bench latency`.

## Generated Local Artifacts

| Artifact | Purpose |
| --- | --- |
| `data/sanity_tp2_ep_latency_sweep.json` | Raw p50 latency for batch sizes 1, 2, 3, 5 |
| `data/sanity_tp2_ep_latency_sweep_b2_verify.json` | Raw p50 latency for batch sizes 4, 6, 10 |
| `data/timings_sanity_tp2_ep_est_f15.csv` | Timing input for `speedup_envelope.py` |
| `data/envelope_sanity_tp2_ep_est_f15_break_even.csv` | Break-even envelope |
| `data/envelope_sanity_tp2_ep_est_f15_target_1p3.csv` | 1.3x target envelope |
| `data/acceptance_sweep_sanity_tp2_ep_est_f15.csv` | Acceptance-rate sweep |
| `logs/*.log` | Raw benchmark logs |

## Measured p50 Latencies

| Batch size | p50 latency (ms) |
| ---: | ---: |
| 1 | 39.475 |
| 2 | 76.741 |
| 3 | 77.508 |
| 4 | 76.639 |
| 5 | 78.525 |
| 6 | 77.027 |
| 10 | 90.365 |

## Timing Envelope Input

| reference_stack | B | k | `t_base_ms` | `t_draft_local_ms` | `t_verify_ms` |
| --- | ---: | ---: | ---: | ---: | ---: |
| sanity_tp2_ep_allgather_est_f15 | 1 | 1 | 39.475 | 33.554 | 76.741 |
| sanity_tp2_ep_allgather_est_f15 | 1 | 2 | 39.475 | 33.554 | 77.508 |
| sanity_tp2_ep_allgather_est_f15 | 1 | 4 | 39.475 | 33.554 | 78.525 |
| sanity_tp2_ep_allgather_est_f15 | 2 | 1 | 76.741 | 65.230 | 76.639 |
| sanity_tp2_ep_allgather_est_f15 | 2 | 2 | 76.741 | 65.230 | 77.027 |
| sanity_tp2_ep_allgather_est_f15 | 2 | 4 | 76.741 | 65.230 | 90.365 |

## Break-Even Envelope

| B | k | `S_max` | `beta_min@1.0x` | Verdict |
| ---: | ---: | ---: | ---: | --- |
| 1 | 1 | 0.716 | NA | no-go |
| 1 | 2 | 0.819 | NA | no-go |
| 1 | 4 | 0.928 | NA | no-go |
| 2 | 1 | 1.082 | 0.849 | go |
| 2 | 2 | 1.110 | 0.898 | go |
| 2 | 4 | 1.092 | 0.956 | go |

## 1.3x Target Envelope

| B | k | `S_max` | `beta_min@1.3x` | Verdict |
| ---: | ---: | ---: | ---: | --- |
| 1 | 1 | 0.716 | NA | no-go |
| 1 | 2 | 0.819 | NA | no-go |
| 1 | 4 | 0.928 | NA | no-go |
| 2 | 1 | 1.082 | NA | no-go |
| 2 | 2 | 1.110 | NA | no-go |
| 2 | 4 | 1.092 | NA | no-go |

## Interpretation

The sanity result matches the expectation from Phase 01:

- Single-node EP has little room for communication-disabled draft.
- Even with the optimistic `f_exposed=0.15` local-draft estimate, `B=1` cannot
  break even under this verify-shape approximation.
- `B=2` can barely break even, but only with high acceptance
  (`beta >= 0.85-0.96`) and cannot approach 1.3x.

This does **not** invalidate the research direction because the target setting
is multi-node DeepEP/DBO decode. It only confirms that single-node sanity is a
weak regime.

## Next Step

The primary Phase 02 experiment still needs a target environment with:

1. a real or dummy large MoE model configured for multi-node DP+EP,
2. DeepEP low-latency available,
3. DBO enabled through a serving or multi-process offline path,
4. timing collection for `deepep_only_verify` and `deepep_dbo_verify`.

Once those timings are collected, rerun the same envelope scripts and compare
against this sanity result.
