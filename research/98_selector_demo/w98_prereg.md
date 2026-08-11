# W98 preregistration — two-round selector demonstration

Date: 2026-08-11

Status: **frozen for every decision that must precede data. Incomplete in one
named respect (the prompt manifest, below), and therefore NO GPU work is
authorized by this document.** Amending anything here after scored data exists
invalidates the claim it supports.

## What preregistration is for here

Both rounds make claims that are only meaningful if the commitments were made
before the measurements. D1 claims the cost model eliminates *soundly*, which
is testable only against a held-out set chosen before it was measured. D2(b)
claims the knapsack beats its controls, which is meaningful only if the
controls were fixed in advance. This document fixes those, and the artifacts
in `data/prereg/` carry the machine-checkable form.

| artifact | sha256 (first 16) |
| --- | --- |
| `data/prereg/w98_prereg_matrix.json` | `de40fa6a8b3a840e` |
| `data/prereg/w98_d1_heldout.json` | `36d07632545ef40d` |
| `data/prereg/w98_d2_controls.json` | `0f7588da4ada90a7` |

## Frozen lattice

- **quant**: `target-matching`, `w4a16-quantized`
- **window**: `128, 256, 512, 1024, off`
- **skip counts**: `0, 4, 8` (0 is skip-off; non-zero counts feed the knapsack)
- **K**: `2, 4`, both read from one **KMAX = 8** position-resolved stream
- **regimes**: `R4, R5, R5cot, R8, R1, R6`
- **content seeds**: `0, 1` (disjoint)

This closes to **30 composed configurations**, of which **8** are single-lever
profiles that Round 1 fits from.

## D1 — Round-1 soundness

Model, fit from **single-lever profiles only**:

```text
D(quant, w, k) ~= (1 - k/L) * ( W_bytes(quant)*kappa_w(x)
                                + KV_bytes(w)*kappa_kv(x) + c0(x) )
```

Claim: 95% intervals for `q_a(x) = P_a(x)/T(x)` cover the held-out **composed**
configurations' measured values, with **zero** false eliminations under
`(K+1)/q_lo < 1 + epsilon_arm`, `epsilon_arm = 0.015`.

**Held-out set — 8 composed triples, frozen now, by a deterministic balanced
fraction rather than sampling.** Four span the weight-dominated corner (window
128/256, where the KV term is small and weights dominate) and four the
KV-dominated corner (window 1024/off). The fraction is balanced so the
held-out check exercises every axis it claims to: each quant path appears
4 times, each non-zero skip count 4 times, each held-out window twice, and
quant × skip is balanced within each corner. The remaining 22 form the fit
set. The exact triples are in `w98_d1_heldout.json`.

A coverage failure creates a **local non-pruning mask** for that stratum (W14
semantics), not a global failure.

### Dependency that narrows D1, recorded now

Phase 96 **W14/D has 21 raw scored boots but no scored surface**
(`research/96_selector_foundations/data/w14/`, verified 2026-08-11). Phase
98's own rule is that W14/D scoring lands before G98-B, *"otherwise the
fallback mode is used and D1 narrows."*

So unless that scoring lands first, the affected strata run Round 1 in
measure-everything mode and **D1 is reported as not exercised there**. This is
registered before data precisely so it cannot later be presented as a design
choice.

## D2 — Round-2 selection quality

- **(a) Screen honesty.** No composition admitted by the product bound
  `f_set >= prod f_i` has a confirmed tau below the bound minus the declared
  tolerance. The bound is **admit-only**: it may let a bad composition through
  to confirmation, never eliminate a good one.
- **(b) Knapsack identity.** At each surviving count `k`, the knapsack set
  beats the **count-matched random-set and worst-set controls** on confirmed
  tau, per regime, outside the tie-set noise. Control seeds are frozen at
  `98001, 98002, 98003` in `w98_d2_controls.json`.
- **(c) u-axis resolution.** `tau(w, g, u)` reproduces the I2 crossing
  structure — window preference flipping with generated-suffix length — with
  **interval separation, not point estimates**.

### Round-2 measurement contract

- one unconditionally-armed **KMAX = 8** stream per (composition, regime):
  greedy, fixed batch shape, pinned realization, per-position reached/accepted
  counters binned by generated-suffix position;
- **2,000–4,000 armed steps** per (composition, regime, u-bucket);
- replication over **content seeds, not boots**, since acceptance is
  near-deterministic under greedy at fixed batch and realization;
- uncertainty by **request-level bootstrap**; identical prompt sets across
  compositions for paired comparison;
- accounting closes as `E + C = A + H` with explicit `H` and clip term `C`;
- **screen then confirm**: singles measured per regime, compositions screened
  by the product bound, only portfolio finalists confirmed with composed
  bursts under the Phase 97 composition contract.

**u-buckets** are provisional at `[0, 256), [256, 1024), [1024, 3072), [3072, ∞)`.
Final boundaries are a **scored output** of the first Round-2 run, placed where
`tau(w, g, u)` actually crosses, then shared verbatim with the runtime policy
key. Registering them as provisional is deliberate: fixing them now would
prejudge D2(c).

**Realization bridge.** Screening may run eager/piecewise. One paired boot
quantifies the eager-vs-captured acceptance bridge; survivors must clear
`tau*` by **more than the bridge width**; finalists are confirmed under the
deployed realization.

## D3 — end-to-end selector value

On the declared workload mix, the selector's chosen configuration per regime
(boot-static composition plus the live K/OFF ladder) achieves at least **90%**
of the omniscient per-segment composite, and its LCB beats every static single
configuration including OFF under the **+2% LCB** rule, in decode currency.

The 90% target follows the Phase 82 precedent (95% of omniscient, 97% of the
unconditional ceiling). It is frozen here and amendable only before data.

D3's composite gap is also the number that sizes Phase 97's G2b/G3/B1 — see
`research/97_composition_runtime/design_p5_runtime_layer_skip.md`, which is
explicitly sequenced behind it.

## Gate order

`G98-0 → G98-A → G98-B → G98-C → G98-D`, no skipping.

- **G98-0** — CPU harness. **Passed**: 56/56 tests, 8 registered proofs, in
  `data/g98_0/w98_harness_validation.json`.
- **G98-A** — smoke: every required boot class initializes and passes
  shared-KV/alias/sampler preflights. ~5 non-scored boots.
- **G98-B** — Round 1 profiles and factored fit; held-out predictions
  committed **before reveal**.
- **G98-C** — Round 2 singles, screen, confirmations, knapsack + controls.
- **G98-D** — end-to-end comparison (D3).

## Assumptions that can still invalidate this matrix

1. **A quantized draft can boot with shared target KV on the current stack is
   UNVERIFIED.** It is a G98-A item. If it fails, the `quant` axis drops out,
   the lattice halves from 30 to 15, and the held-out split must be re-frozen
   before any Round-1 measurement. This is the single assumption most likely
   to change the matrix.
2. Position counters must support KMAX = 8, and u-binning must be
   recorder-side (offline from raw captures) requiring no engine change.
3. The P4 recorder adapts to phase-98 manifests without modifying Phase 97
   frozen sources.

## What is NOT yet frozen — the one gap

**The phase-local prompt manifests do not exist**, so their hashes are not in
this document. Phase 98 requires fresh phase-local manifests over the regime
family with ≥2 disjoint content seeds; Phase 97's generator
(`research/97_composition_runtime/scripts/generate_p4_prompt_manifest.py`) is
CPU-only and deterministic over locally cached datasets and is the natural
basis.

Until a generated manifest's sha256 is committed here, this preregistration is
**incomplete and authorizes no GPU work**, including G98-A. Recording the gap
rather than deferring the whole document keeps the decisions above frozen
ahead of the data they constrain, which is the part that would be worthless if
written later.

## Boundary

Preregistration only. No GPU command is authorized. Every scored run still
requires its own explicit source-bound authorization record, per the rules
carried from Phases 96 and 97.
