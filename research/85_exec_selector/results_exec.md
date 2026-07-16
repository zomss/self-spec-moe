# Execution-extended selector fit


## dense_broken: phi=0.144, psi=0.000
- in-fit |relerr|: max 0.059, median 0.038
- LOO: b32/16k K6 (81-E1 pw): 1.56->1.51 (-3.3%); b32/16k K6 (80-E3): 1.47->1.57 (+6.4%); b32/16k K4 (80-E3): 1.48->1.54 (+3.7%); b8/16k K4 (80-E3): 1.41->1.29 (-8.1%)
- TRACE cross-check: phi_fit 0.144 vs launch-idle/Tt 0.172 vs (step-R*Tt)/Tt 0.129

## dense_fixed: phi=0.000, psi=0.000
- in-fit |relerr|: max 0.263, median 0.035
- LOO: b32/16k K6: 1.91->1.97 (+3.3%); b32/16k K4: 1.85->1.91 (+3.5%); b8/16k K4: 1.64->1.60 (-2.7%); b8/16k K6: 1.52->1.56 (+2.9%); b16/32k K4: 2.77->2.04 (-26.3%); b16/32k K6: 2.33->2.10 (-10.1%)
- TRACE cross-check: phi_fit 0.000 vs launch-idle/Tt 0.014 vs (step-R*Tt)/Tt -0.059

## moe flip band (diagnostic)
- best constant-R fit residuals: ['-26.5%', '+18.7%', '-2.9%'] -- does NOT fit
- implied per-cell R (psi=0.2): [('b4/2k K2', 0.798), ('b8/2k K2', 1.659), ('b32/2k K2', 1.268)] -- R grows with batch: the failure is REALIZATION-side (E2c), which the model correctly refuses to absorb into phi/psi

## Interpretation

1. **The fit is physical, not curve-fit**: phi_broken = 0.144 fitted from
   THROUGHPUT alone lands between the two independent TRACE decompositions
   (0.129 = (step - R*Tt)/Tt; 0.172 = launch-idle/Tt). phi_fixed = 0.000:
   the fit independently declares the 81 chain floor-free.
2. **The 77% delivery number is now an equation output**:
   delivery = (gammaR+1)/(gamma(R+phi)+1+psi) at (gamma=6, R=0.311,
   phi=0.144) = 0.769 — the measured 80-E3 delivery (77%) reproduced from
   two fitted constants. Delivery(gamma,R) has its functional form.
3. **The 32k "misfit" is the flagged modeled-R being exposed**: implied
   R at b16/32k K4 = 0.108-0.14 vs the additive model's 0.28 — consistent
   with the attended-length h-term (the model's known edge, 80-E1) now
   visible on dense. The K6/K4 32k tension sits inside the cells' 10-17%
   run variance; flagged, not fitted.
4. **The MoE flip band correctly REFUSES the phi/psi explanation**
   (residuals -26%/+19%) and returns implied R growing with batch
   (0.80 -> 1.66/1.27) — the E2c realization diagnosis, produced by the
   model instead of by narrative. The extended selector separates floor
   failures from realization failures by construction.

**Selector consequence**: speedup = tau/(gamma(R+phi)+1+psi) with (phi,
psi) per (arch, chain-impl) tier replaces the delivery discount AND the
"priced vs delivered" annotations: an undelivered realization is one
whose (phi, psi, R_realized) have not been measured — the map can now
print LCB-priced numbers per realization. gamma* also shifts: with
phi > 0, optimal gamma drops (the K4~K6 signature is the derivative of
the new denominator).

## Residency feasibility (deterministic; validated on incidents)
| config | resident GiB | feasible | recorded outcome |
|---|---|---|---|
| moe + bf16 full replica, b4/2k | 77.2 | N | matches record |
| moe + bf16 partial 50%, b4/2k | 50.2 | Y | matches record |
| moe + fp8 full replica, b4/2k | 50.2 | Y | matches record |
| dense + W4 draft, b32/32k | 78.5 | N | matches record |
| dense + W4 draft, b16/32k | 50.7 | Y | matches record |
| dense + W4 draft, b32/16k | 51.8 | Y | matches record |

All incident validations PASS. Selector rule: a config is admissible iff feasible(realization, cell); the map prints infeasible cells as such instead of hand notes.

## kappa(M) fit: kappa = 0.28 * max(0, 1 - M/0.2)
| M | implied kappa | model | source |
|---|---|---|---|
| 0.375 | +0.97 | +0.00 | mla ds_fp8block b4/2k [S4-class outlier: b8 neighbor implies ~0] |
| 0.750 | +0.01 | +0.00 | mla ds_fp8block b8/2k [context] |
| 3.000 | +0.12 | +0.00 | mla ds_fp8block b32/2k [context] |
| 0.062 | +0.11 | +0.19 | moe m_fp8block b4/2k (DP4) [noisy: +-10%] |
| 0.125 | +0.08 | +0.11 | moe m_fp8block b8/2k [noisy] |
| 0.500 | +0.10 | +0.00 | moe m_fp8block b32/2k [noisy] |
| 0.062 | +0.21 | +0.19 | 83-E2c b4 realization delta |
| 0.500 | -0.04 | +0.00 | 83-E2c b32 realization delta |

- FIT BASIS (honest): the paired realization deltas (same cell, same beta, kernel-only difference) carry the fit; serve rows are context (weights 0-0.2; the MLA b4/2k implied 0.97 is an S4-class outlier -- its b8 neighbor implies ~0). Re-prediction: the flip inversion (fp8 beats bf16-partial at M>=0.5, loses at M<=0.06) is reproduced by construction of the ramp.
- Scope: fp8-dequant grouped-GEMM kernels; MoE Marlin shows the same sign with smaller k0 (not separately fitted -- data within noise). Selector rule: any fp8 draft realization prices with kappa(M) at its CHAIN M, which is batch- and parallelism-dependent -- constants alone CANNOT price a realization.

## Map v6 (data/strategy_map_v6.md)

LCB selection over the execution-extended model, feasibility-filtered,
600 MC draws per cell. Headlines: (i) dense composed+vres wins every
feasible cell (LCB 1.16-2.00, P(win) 0.66-0.95) — the 84 lm_head find
carried through selection; (ii) the v5 MoE 2k flips are DEMOTED to OFF
by their own uncertainty (LCB 0.89-0.93) — the selector now agrees with
the E2c parity-band measurement; (iii) MLA prints ngram at LCB ~2.45
P1.00 with the e2e-unvalidated flag; (iv) each cell carries winner + LCB
+ P(win) + provenance, and the map emits its own measurement queue.

## v6 measurement queue: both corrections landed, map v6.1 emitted

| queue item | priced (LCB/median) | measured | correction |
|---|---|---|---|
| ngram e2e, mla b32/16k | 2.44 / 2.88 | **0.78x** (accept 2.846) | psi_ngram = 2.4 Tt-units MEASURED: vLLM's proposer does a CPU O(ctx) scan per request per cycle (32x16k here) + verify width. The beta physics transferred (accept ~ block prediction); the REALIZATION is the cost. A GPU-side/суffix-automaton lookup would re-open the cell -- recorded as the lever's realization gap, exactly like flr's replica story. |
| ngram e2e, mla b8/16k | — | 0.77x (accept 2.463) | same |
| flr50 moe b4/16k | 1.15 / 1.35 | **1.00x** (accept 2.943) | FLR_CTXFAC corrected to 1.0: the flr draft is FULL-attention, so its cost grows with ctx like the target's -- R is ctx-invariant. The 16k/32k flr cells revert to window winners. beta stable across ctx (2.94 vs 2.88) as the offline surface said. |

v6.1: MLA returns to OFF on all cells (best LCB 0.88-0.91, now with the
model-free challenger measured too -- OFF is backed by N+1 lever classes);
MoE prints window winners at >=16k and OFF at the 2k band. The selector's
queue-and-correct loop CLOSED within one measurement round: every
provenance flag it emitted was adjudicated, and no confident (high-LCB,
measured-provenance) cell moved.
