# Acceptance-Rate Sweep for Communication-Disabled Draft

Date: 2026-06-23

## Purpose

This note estimates when Self-MoE-spec can add incremental speedup on top of
DeepEP and DBO-style overlap as a function of draft acceptance rate. The goal is
to understand the speedup room before running model-quality experiments.

## Model

For draft length `k = 4`, expected emitted tokens per cycle are:

```text
E(beta, k) = 1 + beta + beta^2 + beta^3 + beta^4
```

The normalized speedup is:

```text
speedup(beta, k) =
    E(beta, k) * T_baseline
    / (k * T_draft_local + T_verify)
```

This sweep uses an optimistic low-batch assumption:

- `T_verify = 1.0`, meaning verification of `k + 1` tokens is close to one
  normal decode step because the step is still mostly latency/memory dominated.
- `T_draft_local = 1 - f`, where `f` is exposed all-to-all fraction after
  DeepEP but before DBO.
- DBO hides 50% of exposed all-to-all, so
  `T_dbo = 1 - 0.5 * f`.

These are normalized estimates, not benchmark results.

## Scenarios

| Scenario | Meaning |
| --- | --- |
| `f = 0.15` | Single-node-like case; exposed communication is small. |
| `f = 0.40` | Multi-node moderate communication exposure. |
| `f = 0.60` | Multi-node high communication exposure. |

## Summary Results

| Scenario | Reference stack | Speedup at beta=0.8 | Speedup at beta=0.9 | S_max | beta break-even | beta for 1.3x |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `f=0.15` | DeepEP-only verify | 0.764 | 0.931 | 1.136 | 0.936 | NA |
| `f=0.15` | DeepEP+DBO verify | 0.707 | 0.861 | 1.051 | 0.975 | NA |
| `f=0.40` | DeepEP-only verify | 0.989 | 1.204 | 1.471 | 0.806 | 0.938 |
| `f=0.40` | DeepEP+DBO verify | 0.791 | 0.964 | 1.176 | 0.919 | NA |
| `f=0.60` | DeepEP-only verify | 1.293 | 1.575 | 1.923 | 0.666 | 0.803 |
| `f=0.60` | DeepEP+DBO verify | 0.905 | 1.103 | 1.346 | 0.851 | 0.983 |

## Interpretation

### Single-node-like exposure is weak

When exposed all-to-all is only around 15%, Self-MoE-spec barely has room even
with perfect acceptance. On top of a DBO-equipped stack, the perfect-acceptance
ceiling is only 1.051x. This is not enough for a strong systems result.

### Multi-node DeepEP verification without DBO has clear room

At `f = 0.40`, the perfect-acceptance incremental ceiling over a DeepEP-only
verification stack is 1.471x, but a 1.3x gain requires `beta >= 0.938`. That is
possible only if local-only routing has very high fidelity.

At `f = 0.60`, the result becomes much more plausible: beta 0.8 is already
near 1.3x over the DeepEP-only stack, and beta 0.9 gives about 1.575x.

### DBO-style overlap is the hard stack to improve

With DBO hiding 50% of exposed communication:

- At `f = 0.40`, Self-MoE-spec cannot reach 1.3x even with perfect acceptance.
- At `f = 0.60`, it can reach 1.3x only with almost perfect acceptance
  (`beta >= 0.983`).

Therefore, adding substantial speedup on top of DBO requires either:

1. very high exposed communication after DeepEP,
2. DBO being ineffective at the target low batch,
3. very high local-only acceptance,
4. cheaper local drafts than full local MoE, or
5. phase-offset scheduling where local draft work helps hide verification
   collectives better than ordinary DBO alone.

## Research Implication

The realistic hierarchy of claims is:

1. **Weak claim:** adds speedup over plain EP or DeepEP without DBO in
   high-communication multi-node decode.
2. **Strong claim:** adds speedup on top of tuned DeepEP + DBO at low-batch
   multi-node decode.
3. **Very strong claim:** adds speedup on top of DBO while also remaining
   training-free and competitive with MTP/EAGLE where those comparisons are
   available.

The next empirical question is whether local-only routing can produce:

```text
beta >= 0.8   for a useful gain over DeepEP-only verification at high f
beta >= 0.9   for a strong gain over DeepEP-only verification at moderate f
beta >= 0.98  for a 1.3x gain on top of DBO in the optimistic f=0.60 case
```

If measured `beta` is far below these thresholds, the project should pivot to
making the draft cheaper or to a hybrid with placement/replication rather than
building a full vLLM runtime path.

Before measuring `beta`, Phase 02 should replace the normalized timing
assumptions in this note with measured `T_base`, `T_draft_local`, and
`T_verify` values for the target reference stacks.
