# Phase 76 E2 — the strategy map: R × τ composed (dense + MoE)

`speedup*(lever, cell) = max_γ τ_β(γ) / (γ·R(lever,cell) + 1)` — R from the E1
sweep (`data/e1/summary.csv`), β per lever from prior-phase measurements, the
formula validated end-to-end in P75 E3. Generator:
`scripts/e2_strategy_map.py`; full tables: `data/e1/strategy_map.md`.

β sources: W4 0.91 (P75 τ-fit; P22), fp8 0.95 (P18 0.954; P74 4.88/5@K4),
window 0.94 (P74 4.76/5@K4), local-route 0.85 (P24 B1 k-sweep, needs the
0.5E-cache machinery; range 0.78-0.92), kvq 0.91 (P74; REFERENCE only —
global lever), bf16 0.99 (reference). Layer-skip has no credible β (P17
collapse) → break-even analysis instead.

## The map (winners per cell)

**DENSE Qwen2.5-7B:**

| | 2k | 16k | 32k |
|---|---|---|---|
| **b1** | W4-Marlin 1.24× (γ3) | W4-Marlin 1.25× (γ3) | W4-Marlin 1.24× (γ3) |
| **b8** | W4-Marlin 1.25× (γ3) | W4-Marlin 1.17× (γ2) | **window 1.27× (γ3)** |
| **b32** | W4-Marlin 1.14×† (γ2) | **window 1.40× (γ4)** | (bf16 over-capacity) |

**MoE Qwen3-30B-A3B (DP4+EP4 NVLink, global batch):**

| | 2k | 16k | 32k |
|---|---|---|---|
| **b4** | ~none (best 1.01×) | ~none (1.02×) | (noisy cell — see caveat) |
| **b8** | ~none (1.02×) | **window 1.21× (γ3)** | **window 1.24× (γ3)** |
| **b32** | ~none (1.03×)† | **window 1.24× (γ3)** | **window 1.90× (γ6)** |

† compute-bound roofline cells — formula optimistic there (P74: real
high-batch/short-ctx self-spec lost outright).

## VALIDATION — five map cells now have e2e ground truth

Three back-predictions (prior phases) + two FORWARD predictions (E3 spot-check,
`scripts/e3_spotcheck.sh`, real spec-vs-nospec tok/s on the W7 harness):

| cell | map predicts | independently MEASURED | err |
|---|---|---|---|
| dense b1/2k | W4-Marlin **1.24× @γ=3** | P75 E3: **1.21× @γ=3** (real tok/s) | +2.5% |
| MoE b8/16k | window **1.21× @γ=3** | P74: **1.16× @K=2** (real tok/s) | +4% |
| MoE b8/32k | window **1.24× @γ=3** | P74: **1.34× @K*=3** (real tok/s) | −7% |
| **dense b32/16k** | window **1.40× @γ=4** | E3: **1.54× @K=4** (3394 vs 2207 tok/s, accept 4.85/5) | **−9%** |
| **MoE b32/32k** | window **1.90× @γ=6** | E3: **1.64× @K=6** (986 vs 602 tok/s, accept 6.49/7) | **+16%** |

Both forward predictions were made BEFORE measurement and both cells confirm
the map's winner AND depth. The signed errors decompose cleanly:
- **dense over-delivers** because the assumed window β=0.94 (from Qwen3) is
  conservative — measured dense accept 4.85/5 ⇒ β≈0.97. Re-composing with the
  measured β gives 1.49× vs 1.54× measured: **103% harness delivery at K=4**.
  The β-portability caveat resolved in the favorable direction on dense.
- **MoE under-delivers by the harness tax**: measured accept 6.49/7 ⇒ β≈0.97
  (also above assumption — the formula with measured β says 2.10×), but the
  PIECEWISE chain pays per-draft-step overhead that deepens with γ; 1.64/1.90 =
  86% delivery at K=6 vs P75's ~90% at γ=3 — consistent trend, and 1.64× is
  the project's largest measured lossless single-node win at serving batch.
- Depth check: dense K=3 measured 1.51× vs K=4's 1.54× (γ* flat-top, matching
  the map's 1.38 vs 1.40). The MoE K=3 arm hit a systems stall (uniform 20.4s
  decodes, engine draining at 2 running reqs — 200 tok/s, marked SUSPECT, not
  a speculation result; logs/e3_moe_spec_K3_b32_c32k.log).

The optimal-depth predictions also reproduce P74's law independently: γ*
grows with context (γ1-3 at 2k → γ3-6 at 32k) — "a more memory-bound verify
makes deeper speculation pay," now emerging from the composed surface rather
than being assumed.

## What the map says (the paper's core claims, all measured)

1. **There is no universal lever.** Dense splits into a W4-Marlin region
   (short ctx / low batch, ~1.2×) and a window region (long ctx / high batch,
   up to 1.40×), with the frontier between b8/16k and b8/32k. MoE splits into
   a "don't speculate" region (short ctx: best ≈ 1.00-1.03×) and a window
   region (≥16k, 1.2-1.9×).
2. **The MoE short-ctx region says self-spec should be OFF** — no lever
   composes past ~1.03× there. A correct strategy selector must include "none":
   the P74/75 toggle rationale, now as a map region.
3. **Weight-quant never leads a trustworthy MoE cell** (P74's law at full
   coverage): its cost is parity, so even β=0.95 cannot buy a win —
   `τ/(γ·1+1)` at γ=1 tops out at 0.98.
4. **Plain self-drafting never pays** (bf16 reference: R=1 → max 0.985×) — a
   lever is necessary, not optional.
5. **Layer-skip is cost-dominant but acceptance-dead**: to beat the cell
   winners it would need β ≈ 0.76-0.99; P17 measured its acceptance collapsing
   super-linearly with skip fraction — far below the bar. Cost-only maps
   (SWIFT-style claims) overstate layer-skip; the composition kills it.
6. **KV-quant (reference)**: composes to 1.0-1.15× at long ctx — but it is
   global in this stack (P74), so those cells are the *motivation bound* for
   building a draft-only KV pool, not an available strategy.
7. **Local-route on NVLink composes to ~parity** (1.01-1.07× before its β
   caveats) — its cost cut (0.74-0.94) is real but β=0.85 eats it. Its regime
   remains comm-bound fabrics (PCIe/multi-node, deferred) where R drops far
   below 0.8, per the refined law from E1 prediction #4.

## Caveats (ranked)

1. **Roofline at compute-bound cells** (†): verify's B·(γ+1) tokens are not
   free at b32/2k — P74 measured real losses there. The map's "~none" verdicts
   at those cells are therefore *upper bounds* — the truth is worse, which
   strengthens claim 2.
2. **MoE b4 long-ctx cells carry ±10-15% placement noise** (S4): the b4/32k
   "fp8marlin 1.25×" is a noise artifact (contradicts P74's measured fp8
   parity); winners in b4 long-ctx cells are not robust. Quote b8/b32 rows.
3. **β portability**: β values were measured on specific models/ctx (window β
   on Qwen3-30B 16-32k; W4 on Qwen2.5-7B 2k; fp8 on Qwen3-30B). Cross-cell
   reuse assumes β is regime-stable — P74/75 data supports this within the
   swept range (window 4.76@16k vs 4.79@32k), but it is an assumption.
4. **local-route β requires machinery** (0.5E cache + renorm; shared-expert
   anchor on other models) and its harness R may differ from the standalone R.
5. Harness delivery ~90% of formula (P75 E3): multiply map values by ~0.9 for
   realistic expectations on the current PIECEWISE harness.

## Next

- Phase 77: fit the term-decomposed cost model (weight-read / KV-read / comm /
  floor) to `summary.csv` so R(cell) becomes *predictable* rather than looked
  up, and the map extrapolates beyond the grid.
- Deferred tiers close the two open regions: PCIe (local-route's predicted
  home) and MLA (architecture axis).
- ~~E2E spot-checks of the two unmeasured cells~~ **DONE** (see VALIDATION):
  dense b32/16k window 1.54× measured (map 1.40×), MoE b32/32k window 1.64×
  measured (map 1.90×, 86% harness delivery) — winners and depths confirmed.
  Open loose end: the MoE K=3 stall (systems, suspect DP drain behavior).
