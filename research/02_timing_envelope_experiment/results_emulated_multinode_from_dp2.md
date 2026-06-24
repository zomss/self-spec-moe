# Results: Emulated Multi-Node Envelope from GPU-6/7 DP2 Data

Date: 2026-06-23

## Status

This is an **emulation**, not a real multi-node benchmark.

It reuses the measured verification-shape scaling from
`results_dp2_gpus6_7.md`, then injects synthetic exposed-communication
fractions to approximate multi-node regimes.

## Emulation Model

Input data:

- measured DP2+EP allgather timings on GPUs 6 and 7,
- global batch `B = 2`,
- draft lengths `k = 1, 2, 4`,
- measured verification-shape ratio:

```text
r_verify(B, k) = measured_latency(B * (k + 1)) / measured_latency(B)
```

For exposed communication fraction `f`, the emulator uses:

```text
T_base_deepep = 1
T_draft_local = 1 - f
T_verify = r_verify
```

For DBO, the emulator uses hide fraction `h = 0.5` and reports two variants:

1. **Reference-only DBO:** DBO improves the reference stack but not the
   Self-MoE-spec verification step.
2. **Verification-overlap DBO:** DBO also applies inside the exact verification
   step, treating DBO as orthogonal to Self-MoE-spec.

These two variants bracket the sensitivity to how well DBO composes with the
verify phase.

## Generated Artifacts

| Artifact | Purpose |
| --- | --- |
| `data/timings_emulated_multinode_from_dp2_gpus6_7.csv` | Emulated timing input |
| `data/envelope_emulated_multinode_from_dp2_break_even.csv` | Break-even envelope |
| `data/envelope_emulated_multinode_from_dp2_target_1p3.csv` | 1.3x target envelope |
| `data/acceptance_sweep_emulated_multinode_from_dp2.csv` | Acceptance-rate sweep |

## Key Result for `B=2, k=4`

| Reference stack | `S_max` | Speedup at beta=0.9 | `beta_min@1.3x` |
| --- | ---: | ---: | ---: |
| DeepEP-like, `f=0.20` | 1.192 | 0.976 | NA |
| DeepEP+DBO ref-only, `f=0.20` | 1.073 | 0.879 | NA |
| DeepEP+DBO verify-overlap, `f=0.20` | 1.099 | 0.900 | NA |
| DeepEP-like, `f=0.40` | 1.473 | 1.206 | 0.938 |
| DeepEP+DBO ref-only, `f=0.40` | 1.178 | 0.965 | NA |
| DeepEP+DBO verify-overlap, `f=0.40` | 1.251 | 1.025 | NA |
| DeepEP-like, `f=0.60` | 1.927 | 1.578 | 0.802 |
| DeepEP+DBO ref-only, `f=0.60` | 1.349 | 1.105 | 0.982 |
| DeepEP+DBO verify-overlap, `f=0.60` | 1.524 | 1.248 | 0.921 |

## Interpretation

The emulated multi-node result matches the Phase 01 conclusion:

- If exposed communication is modest (`f=0.20`), Self-MoE-spec is not compelling.
- If exposed communication is moderate (`f=0.40`), there is useful room over a
  DeepEP-like stack, but a 1.3x gain requires very high acceptance
  (`beta ~= 0.94` for `k=4`).
- If exposed communication is high (`f=0.60`), the method has meaningful room:
  beta 0.8 can reach about 1.3x over DeepEP-like verification, and beta 0.9
  reaches about 1.58x.
- DBO remains the hard composition case. If DBO only improves the reference
  stack, reaching 1.3x at `f=0.60` requires almost perfect acceptance
  (`beta ~= 0.98`). If DBO also overlaps the verify phase, the requirement
  improves to `beta ~= 0.92`.

## Research Implication

The conclusion is **qualified yes**: multi-node EP is the right target regime,
and Self-MoE-spec can be effective there, but not automatically. The deciding
conditions are exposed communication fraction and local-only draft acceptance.

The two central challenges are:

1. Demonstrate that exposed all-to-all communication remains high in the target
   low-batch multi-node EP decode stack, even after DeepEP and DBO/overlap.
2. Demonstrate that local-only draft acceptance is high enough to realize the
   communication envelope.

A third systems challenge is ensuring that DBO or phase-overlap composes with
the exact verification step instead of reintroducing all-to-all into every draft
step.

## Draft Length Assumption

The current headline emulation uses `k = 4` draft tokens per verify cycle. The
Phase 02 sweeps cover `k = 1, 2, 4`, and the planned measured sweep should also
include `k = 8`.

`k = 4` is the default first target because it is large enough to amortize one
expensive verification collective, but not so large that local-routing error and
verification payload are likely to dominate immediately. The expected operating
point should be selected by sweeping:

```text
k = 1, 2, 4, 8
```

and choosing the smallest `k` that gives enough collective amortization at the
measured acceptance rate.

For a strong paper result, the target should be:

```text
multi-node decode where exposed communication after DeepEP is high
+ local-only acceptance beta >= 0.9
+ DBO/phase-overlap composes with exact verification
```

If measured local-only acceptance is below 0.8, the project likely needs a
cheaper draft design or stronger placement/replication before runtime work.

The claim should therefore be:

> Self-MoE-spec is promising for low-batch multi-node EP decode when exposed
> all-to-all remains high and local-only draft acceptance is sufficiently high.

It should not be stated as "multi-node EP always makes Self-MoE-spec effective."

## Caveats

- This is not a substitute for real multi-node DeepEP/DBO timing.
- The verification-shape ratios come from single-node DP2 allgather, not
  inter-node DeepEP.
- DBO composition is modeled, not measured.
- Absolute latencies from the local DP2 run are not meaningful for multi-node;
  only shape ratios are reused.
