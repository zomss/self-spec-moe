# Phase 78 — the term-decomposed cost model: R becomes predictable, not looked up

Source phases: 76 (measured R surface `data/e1/summary.csv` — the fit input;
E1 found the crossovers this model must reproduce), 77 (measured β
`data/beta.csv` — the FIXED accept side; strategy map v2), 74/75 (the
qualitative law: "a lever wins iff it cuts the binding term"), 72/73 (the
fixed kernel-launch floor), 24 (comm-term physics; PCIe recipe, deferred).

## Objective

Fit a small physical model of the decode step so that `R(lever, cell)` — and
therefore the strategy choice — can be PREDICTED for cells, levers, and
architectures that were never swept:

```
T(model, arm, b, ctx) =
    F(b)                                  # fixed/launch floor (P72: ~2400-kernel
                                          #  floor at b1; DP-coord adds a per-
                                          #  group constant — moe b4 7.65ms vs
                                          #  dense b1 5.98ms at same work)
  + [ W_bytes(arm) + KV_bytes(arm, b, ctx) ] / BW_eff
  + C_comm(b) · 1[EP fabric]              # a2a dispatch+combine
  + κ_kernel(arm)                         # per-kernel overhead (Marlin dequant
                                          #  tax etc. — E1: Marlin ≠ Machete)
```
with a **max(memory, compute) roofline variant** as the alternative form for
the compute-bound corner (b32/2k) — fit both, keep whichever validates.

Levers enter ONLY through their term edits (this is the whole point):
weight-quant scales `W_bytes` (×0.5 fp8, ×0.25 int4) + its κ_kernel; window
caps `KV_bytes` at `min(ctx, W+sinks)`; kv-quant halves `KV_bytes`; skip
multiplies all per-layer terms by (1−s); local-route zeroes `C_comm` and cuts
remote-token expert compute. Architecture enters ONLY through known constants
(weight bytes, KV bytes/token, layers, active-expert bytes) — so the model
extrapolates to a NEW architecture from its config alone.

## Identifiability (why the fit is possible without the PCIe tier)

The user deferred the fabric axis; within one fabric the comm coefficient is
still identified because **the lever arms are term-isolating probes**:
localroute/skip-a2a deltas vs bf16 measure `C_comm` (+remote-compute) at
NVLink directly; window/kvq deltas trace the KV term against ctx×batch; quant
deltas trace the weight term; skip scales everything at once (a consistency
check across terms); bf16 cells anchor `F(b)` and `BW_eff`. The PCIe tier,
when it runs, becomes a pure out-of-sample test of the fitted `C_comm` scaling
rather than a fitting requirement.

## Fit + validation ladder (pre-registered)

- **V0 in-grid**: fit on 76's summary.csv absolute TPOTs (dense 8 arms,
  moe 9 arms × 9 cells minus over-capacity). Gate: median |rel err| ≤ 5%,
  and the model must REPRODUCE the two measured crossovers (dense W4→window
  along ctx×batch; MoE OFF→window along ctx) — a model that fits TPOT but
  misplaces the crossovers fails the phase.
- **V1 held-out-arm**: refit with the window arms EXCLUDED; predict window's
  R curve purely from KV-byte accounting. Gate: ≤ 10% on the held-out curve.
  (Repeat holding out quant.)
- **V2 forward, cheap**: predict R(win128) — 77 measured its β but 76 never
  measured its COST; and predict two never-swept interpolation cells (b16 ×
  8k). Then run exactly those E1 cells (one arm, ~30 min). Gate: ≤ 10%.
- **V3 forward, the payoff**: predict the ENTIRE deferred MLA cost tier from
  DeepSeek-V2-Lite's config constants alone (weight 31 GB, KV 30 KiB/tok,
  27 layers) BEFORE running it; then run `76/scripts/e1_sweep.sh mla` as the
  ground truth. Gate: ≤ 15% median, crossovers correctly placed. Passing V3
  is the paper's claim: *the selector works on an architecture it never swept*
  — MLA's β is already measured (77), so V3 completes MLA's strategy-map
  column with only ONE confirmation sweep.
- **V4 (composition)**: `selector.py` = R̂ (model) × β (77 table) → predicted
  best (lever, γ) per cell; check against strategy map v2's winners and the
  five e2e ground-truth cells (within their known delivery factors).

## Method / commands (to be built)

- `scripts/fit_cost_model.py` — loads 76 summary.csv (+ model constants
  table), fits additive and roofline forms per group (shared physical
  parameters where justified: BW_eff per GPU, F per engine mode), writes
  `data/fit.json` + per-cell residuals.
- `scripts/predict.py --model <cfg> --arm <lever> --b --ctx` — R̂ from config
  constants (V2/V3 driver).
- `scripts/selector.py` — R̂ × `77/data/beta.csv` → strategy per cell; emits
  the predicted MLA strategy-map column (V3) and map v3 after validation.
- Boxes/GPUs: fitting is CPU; validation sweeps reuse 76's runner on GPUs
  0/1/6/7 (env_e76.sh; box constraints in memory).

## Assumptions / risks

1. Serve-mode TPOT decomposes additively (or as a two-term roofline) — the
   DP4 placement noise (76: median-of-4, ±10% at b4 long-ctx) bounds
   achievable fit quality on MoE; weight cells by their run spread.
2. κ_kernel is arm-specific but cell-INDEPENDENT — E1's Marlin batch trend
   suggests mild M-dependence; if V0 residuals show structure vs b, promote
   κ to κ(b) with one slope.
3. MLA's serve stack differences (FLASH_ATTN_MLA, decompressed vs latent KV
   read in vLLM) may add an unmodeled per-arch term; V3's 15% gate absorbs
   modest mismatch, and a failure is itself the E4-style refinement.

## Decision criteria

Phase SUCCEEDS if V0-V2 pass and V3 places MLA's crossovers correctly
(magnitudes within 15%). Deliverable claim: a selector that, given a model
CONFIG and a (batch, ctx) operating point, predicts the best self-spec lever
and its speedup without sweeping — validated on an unswept architecture.

## Expected next artifact

`results_cost_model.md` + `data/fit.json` + `scripts/{fit_cost_model,predict,
selector}.py` + strategy map v3 (dense + MoE + MLA column). Then paper
assembly (with the duplicate-work check on the specific claim vs
MagicDec/QuantSpec/SWIFT per the repo contribution policy).
