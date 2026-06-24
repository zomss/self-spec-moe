# Phase 08: Negative-Result Write-up

Source phases: [`../01_communication_advantage`](../01_communication_advantage)
through [`../07_recent_model_acceptance`](../07_recent_model_acceptance)

## Objective

Synthesize Phases 01-07 into a single scoped negative result for Self-MoE-spec
(local-only routed draft + exact full-MoE verification), now that the acceptance
evidence spans old/recent and fine/coarse MoE checkpoints.

This is a synthesis phase, not a new experiment. The decision criteria from the
earlier phases (`beta >= 0.8-0.98` under realistic placement) were not met, and
Phase 07 added the structural anti-correlation finding, so the project moves to
write-up.

## Deliverable

| Output | Path |
| --- | --- |
| Negative-result report | `negative_result_report.md` |

## Core claims

1. Local-only drafting reaches at best ~0.50 (Qwen3-30B) / ~0.70 (GPT-OSS-20B)
   one-token sampled acceptance at EP2, and far less at EP8 — never the required
   `beta >= 0.8`.
2. Acceptance and the exposed-communication advantage are **anti-correlated**
   along the EP axis: acceptance is best at low EP (small advantage) and collapses
   at high EP (the only regime with a real advantage).
3. Scope: all tested models lack a shared expert. A large always-local shared
   expert (EP-invariant mass) is the one untested mechanism that could overturn
   this; it is named as future work, not tested here.

## What would reopen this

Run the Phase 07 acceptance proxy on a shared-expert MoE (DeepSeek / Llama-4 /
GLM style) and check whether high-EP acceptance stays useful. See
`negative_result_report.md` Section 8.
