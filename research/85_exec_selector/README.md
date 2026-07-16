# Phase 85 — the execution-extended selector

Source: reviewer pass 2026-07-17 (method critique #1: delivery is a law
with no equation), the full e2e record (80-E3, 81-E1/E3, 83-E2c, MoE
check, MLA challenger), 81-E0 traces.

## The model

  speedup(γ) = τ_β(γ) / ( γ·(R + φ) + 1 + ψ )

R = the map's draft step ratio (standalone bytes/compute); φ = per-draft-
step fixed floor; ψ = per-cycle boundary overhead — both in units of the
target's decode step T_t, fitted ONCE per (architecture, chain
implementation) from measured e2e cells. The old selector is the φ=ψ=0
limit. Delivery(γ,R) becomes arithmetic: delivery = (γR+1)/(γ(R+φ)+1+ψ).

## Validation design

1. Fit (φ, ψ) per chain config by least squares on the e2e record
   (dense broken chain: 4 cells; dense fixed chain: 6 cells).
2. Leave-one-out residuals (the fit must predict held-out cells).
3. **The physicality cross-check**: φ_fit vs the INDEPENDENT trace
   anatomy (81-E0: launch idle 2.53 ms/step broken, 0.21 ms fixed) —
   fit-from-throughput must reproduce trace-from-kineto.
4. Diagnostic power: the MoE flip band (b4 delivered / b8+b32 blocked)
   should NOT fit a constant-R φψ model — its residuals must point
   R-side (the E2c realization reading), demonstrating the model
   separates floor failures from realization failures.

Then: κ(M) per-kernel function (the fp8 small-M penalty, two recorded
mispricings) and the residency feasibility constraint — map v6 inputs.

## Artifacts

scripts/fit_exec.py -> data/exec_fit.json + results_exec.md.
