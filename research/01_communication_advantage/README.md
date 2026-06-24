# Phase 01: Communication Advantage Envelope

Source phase: [`../00_proposal`](../00_proposal)

## Objective

Pinpoint the best-case advantage of communication-disabled local MoE drafting
before implementing local routing in vLLM or measuring real draft acceptance.

The phase asks:

```text
Given measured T_base, T_draft_local, and T_verify,
how high must local-only draft acceptance be to add speedup on top of
DeepEP and DBO-style reference stacks?
```

## Scope

This phase treats DeepEP and DBO as orthogonal components:

- DeepEP is the verification communication backend.
- DBO or DBO-like overlap can be applied inside verification or at phase
  boundaries.
- Self-MoE-spec adds local collective-free draft cycles on top of that stack.

## Assets

| File | Purpose |
| --- | --- |
| `speedup_envelope.py` | Computes `S_max`, `beta_min`, and acceptance-rate sweeps from timing CSVs. |
| `acceptance_sweep_analysis.md` | Normalized acceptance sweep for single-node-like and multi-node-like exposure. |

Keep this phase focused on the calculator and normalized analysis. Measured
timing inputs, logs, plots, and result tables belong to Phase 02.

## Commands

Break-even or target-speedup envelope:

```bash
.venv/bin/python research/01_communication_advantage/speedup_envelope.py \
    timings.csv --target-speedup 1.3
```

Acceptance-rate sweep:

```bash
.venv/bin/python research/01_communication_advantage/speedup_envelope.py \
    timings.csv --acceptance-rates 0.5,0.6,0.7,0.8,0.9,1.0
```

Input CSV schema:

```csv
reference_stack,batch_size,draft_length,t_base_ms,t_draft_local_ms,t_verify_ms
deepep_only_verify,1,4,1.0,0.6,1.0
deepep_dbo_verify,1,4,0.8,0.6,1.0
```

## Decision Criteria

Proceed to Phase 02 measured timing experiments only if this normalized envelope
shows plausible room:

- useful gain over DeepEP-only verification at plausible `beta`,
- measurable incremental gain on top of DeepEP+DBO or phase-overlap,
- no dependence on disabling or misconfiguring orthogonal communication kernels.

If the envelope is weak, pivot to making drafts cheaper, improving placement, or
using phase-offset scheduling before investing in a full runtime path.

## Next Artifact

The next artifact is the measured experiment plan in
[`../02_timing_envelope_experiment`](../02_timing_envelope_experiment).
