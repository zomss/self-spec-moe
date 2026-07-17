# §6 The selector (draft)

> Source: Phase 78 (`results_cost_model.md`, `data/fit.json`,
> `strategy_map_v3.md`) + P75 delivery calibration. Numbers final. ~1.5 pages.

## 6.1 From two surfaces to a decision

The selector composes the two measured surfaces into expected speedup,

  speedup(ℓ, γ; b, c, arch) = τ_β(γ) / (γ·R + 1),

where τ_β(γ) is the expected accepted length from per-token acceptance β
(geometric composition, anchor-calibrated in §5 to ≤2% on both
architectures; delivery ~90% at γ=3, ~86% at γ=6 on the pre-§8 chain), and
R comes from §4 directly where measured. For unmeasured cells and levers, R
comes from a term-decomposed cost model fitted to §4's absolute TPOTs:

  T(b, c) = F(b) + [W_read + KV_read]/BW_eff + h·b·c + C_comm(b) + κ_kernel,

in which levers are TERM EDITS (window shrinks KV_read; W4 shrinks W_read;
skip scales the per-layer terms; local-route shrinks C_comm and the remote
expert read) and architectures are CONSTANTS (F, BW_eff, h, comm
coefficients, per-kernel κ). Two of the constants were discoveries, not
assumptions: the KV-management term h·b·c — draft-side read levers do not
remove block-table/metadata work that scales with ALLOCATED context, which
a pure byte model under-predicts by 33–49% on exactly the large-KV MoE
cells — and the per-kernel κ, which recovers §4's Marlin/Machete ordering
(κ_machete = 0.67 ms, κ_marlin ≈ 0) from the fit alone.

## 6.2 The validation ladder

We validate the model the way it will be used: predicting things it was not
fitted on, with gates registered before measurement.

| rung | question | result | gate |
|---|---|---|---|
| V0 in-grid | does the decomposition fit? | dense median err 3.5%, 8/8 crossover orderings; MoE 6.8%, 8/9 | dense PASS; MoE at its DP4 noise floor |
| V1 held-out lever | predict window's full R curve with ALL window data excluded | **2.2% dense / 6.9% MoE** | **PASS — the load-bearing rung** |
| V2 forward | price a never-measured arm (win128: 2.9%, 17 cells) and a never-swept cell (b16×8k: 8.0%) | registered → verified | PASS |
| V3 transfer | predict the MLA tier from config constants alone | R err 15.5% | **FAIL — reported** |
| V3b one-anchor | one 5-minute bf16 anchor cell; everything else transferred | **R 9.3% on 53 held-out cells**; TPOT 18.3% | R PASS / TPOT FAIL |

V1 is the claim that the decomposition is physics rather than
curve-fitting: window's cost curve is recoverable from KV-byte accounting
alone. The V3→V3b pair is the honest transfer statement: the R-SELECTOR
transfers to an unswept architecture with a single anchor measurement —the
V3 failure decomposed into a step-fixed vs per-layer floor split (the skip
arms prove it: measured skip R is 0.58–0.92, not the 0.50 a
layer-proportional floor implies) and architecture-specific κ at tiny GEMM
M — while absolute TPOT additionally needs 2–3 per-stack constants (h,
BW_eff) that are properties of the serving stack, not the architecture.
We report both halves; the selector only needs the half that passes.

Caveats we carry rather than hide: BW_eff fits above the hardware peak
(4.8 vs 3.35 TB/s) because weight reads partially overlap compute — the
model is EFFECTIVE, validated by prediction, and makes no bandwidth claim;
MoE's V0 residual concentrates in cells flagged for DP4 placement
bimodality in §4.

## 6.3 The strategy map and its e2e check

Composing measured R (dense, MoE, and the anchored MLA tier) with §5's
per-architecture β yields the full strategy map — (architecture, batch,
context) → lever + depth, including OFF:

- **Dense**: a W4 region (short ctx / low batch, 1.28–1.30×) and a window
  region (long ctx / high batch, up to 1.56× single-lever), with window-128
  tying window-512 — §5-F8's size-insensitivity surfacing in the decision.
- **MoE-GQA**: OFF-or-marginal at 2k (≤1.06× — §4-F2's parity band composed
  with β); a window region from 16k (1.27–2.22× roofline).
- **MLA**: OFF everywhere at NVLink — the doubly-weak window (§5-F8) and
  the anchored cost tier agreeing. MLA's measured β strengths
  (shared-expert local-route 0.95–0.99, shallow skip 0.97) are parked for
  the comm-bound fabric (§9).

**OFF is search-backed, not lever-exhaustion.** "OFF" would be a naive
verdict if it meant only that the levers we tried lose — some COMBINATION
might win. We therefore re-derive every OFF cell by exhaustive priced
search over the full combination space (53 configs per cell: all subsets
up to size 4 with ≤1 lever per class, including MLA's best acceptance
lever lr50 β 0.95–0.99, × γ ≤ 8), priced OPTIMISTICALLY — product-law β is
an upper bound (§7's measured exceptions are destructive), measured
combo/single R where available, and delivery = 1 (the §8 chain). Even this
upper bound stays ≤ 1.13× in every OFF cell, and — the structural finding
— NO composed configuration materially beats the best single lever in any
OFF region: composition cannot rescue OFF, because these regions are OFF
precisely where the base cost terms the levers cut are already small
(MLA's compressed KV mutes window; EP amortization mutes weight-quant;
NVLink mutes local-route). The residual challengers are all the portable
fp8 single lever at 1.05–1.13× optimistic; we report them as measured
marginal cells, not OFF flips. [Artifact: off_hardening.md]

All five end-to-end ground-truth cells match the map's winner (5/5), and
the map-vs-measured speedups agree to 2.5–16% pre-§8 (the residual being
delivery, which §7 diagnoses and §8 eliminates). Borrowed-β ablation:
recomposing the map with literature β values gets the REGIONS right and
the values wrong (dense b32/16k: 1.40× borrowed → 1.55× measured vs 1.54×
e2e) — measuring β is what makes the map quantitative.

**Selection under uncertainty (the final selector).** The map of record
(v6) selects by LOWER CONFIDENCE BOUND under per-source uncertainty
(measured R ±4%, term-model R ±10–30% by family, product-β ±0.02,
per-chain-tier execution constants φ/ψ from §8's fits), with residency
feasibility as an admission filter, Monte-Carlo P(winner) per cell, and
γ capped per lever semantics. Three properties follow by construction:
an under-measured realization cannot win (its wide bounds sink its LCB);
every cell prints winner + LCB + P(win) + provenance; and the map emits
its own measurement queue — the cells where uncertainty is
decision-relevant. In its first round that queue adjudicated all three
of its items against the offline layer and the offline layer lost twice
on realization constants and once on our own gate circularity (§7.5) —
without any confident cell moving.

The deployment recipe this section justifies: one R sweep per serving
stack (§4), one β column per (lever, architecture) (§5, ~1 GPU-hour), one
anchor cell per new architecture (5 minutes) — then every regime cell of
the map is a prediction, not a measurement.
