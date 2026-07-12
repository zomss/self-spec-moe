# Phase 80 — lever composition + the offline-profiling search (strategy map v4)

Source phases: 78 (term-decomposed cost model — composes term edits
analytically; selector), 77 (β scorer + stage-A refs on disk — arms STACK;
measured β strength axes), 76 (E1 runner stacks serve flags for combo cost
arms), P23/P25 (prior composition evidence: quant errors "barely compound";
shared+local+FP4 composed as β ≈ min(coverage, quant-floor)).

## Objective

Move from "pick 1 of 6 levers" to "assemble the best DRAFT CONFIGURATION"
per (architecture, batch, ctx): lever SUBSET × strengths × γ — found by
search over measured physics, validated the 78 way (predict → register →
verify only the argmax).

## The method IS the pruning/mixed-precision playbook (deliberately)

This is the same problem shape as structured pruning and mixed-precision
quantization search (SLEB/ShortGPT-style layer removal, HAWQ/AMC-style
bit allocation): **offline profiling of per-component cost and sensitivity +
a composition assumption + search + top-K validation.** Mapping:

| pruning/MPQ concept | ours |
|---|---|
| per-layer latency table | 78's term-edit cost model (R̂ analytic, composes free) |
| per-layer sensitivity (perplexity on calib set) | 77's offline β (teacher-forced; the ACTUAL objective, not a proxy) |
| additivity assumption + its failure | β-composition law (product vs min) — E0 measures WHERE it fails |
| greedy/ILP under budget | small space (~10³/cell) → exhaustive over R̂×β̂; greedy re-profiling reserved for the per-layer extension (E4) |
| final eval | e2e spot-check of registered argmax cells |

Two advantages over the pruning setting: our sensitivity metric (β) is the
objective itself, not a perplexity proxy; and every candidate's cost is
predicted by a VALIDATED model (78 V1/V2), so search never touches a GPU.
Known relative: **SWIFT** does runtime Bayesian search over skip-LAYER-SETS
(one lever); our delta is multi-lever assembly with a measured composition
law — cite it, and reuse its insight that layer choice matters (E4).

## Experiments

- **E0 — the β-composition law** (offline, refs on disk, ~1 GPU-day):
  pairwise matrix on all 3 models — win×quant, win×skip, quant×skip,
  win×kvq, ×local-route (MoE/MLA), + the 2-3 promising triples. Test
  β_combo against (a) product β_a·β_b, (b) min(β_a, β_b). Deliverable: which
  pairs are composable, per architecture (the composition analogue of 77's
  portability verdict).
- **E1 — combo cost validation** (~2 h GPU): R̂_combo from 78's term edits vs
  measured combo arms in the 76 runner (serve flags stack: quant ckpt +
  window override; dummy+layers+window for skip combos). Gate: ≤10% like 78-V2.
  Watch kernel-conflict edge cases (which κ a combo rides).
- **E2 — the search → strategy map v4**: enumerate (subset, strengths, γ≤8)
  per cell with R̂ (78) × β̂ (77 + E0 law); REGISTER the argmax and its
  predicted speedup per cell before any measurement. Include OFF; flag
  roofline cells (γ≥6 delivery ≤86%).
- **E3 — e2e spot-check** of the two largest registered combo wins
  (candidates by prediction: dense b32/16k window+W4 → R̂≈0.35, ≈2× roofline;
  MoE b32/32k window+local-route — the original project thesis with both
  terms measured; MLA shallow-skip+quant — first entry into its OFF region).
- **E4 (stretch) — per-layer assembly**: the literal pruning analogy — profile
  per-layer β sensitivity (which layers tolerate window/skip/quant), greedy
  per-layer lever assignment with offline re-scoring after each addition
  (interaction mitigation, as iterative pruning does). SWIFT covers
  skip-only; multi-lever per-layer drafts are unexplored.

## Pre-registered hypotheses

1. β composes ≈ multiplicatively for levers with INDEPENDENT error sources
   (window×quant, quant×kvq); P23/P25 support this.
2. window×skip is the destructive pair (windowed forward through fewer
   layers compounds super-multiplicatively).
3. The composition law is architecture-dependent in the same direction as 77
   portability (MLA most skip-tolerant → most composable with skip).
4. Best composed configs beat map-v3 winners by ≥1.2× at the long-ctx cells,
   and MLA's OFF region gets its first >1.1× entry (skip125+q_fp8).
5. γ* deepens under composition (cheaper draft → longer chains) — pushing
   into the roofline zone; E3 measures the delivery there.

## Decision criteria

- E0 law holds (≤0.02 β error for the composable pairs) and E1 ≤10% →
  the search is trustworthy; E2/E3 proceed.
- Registered E3 argmax cells deliver ≥85% of prediction → v4 stands.
- Destructive pairs / law failures → documented as the composition verdict
  (the phase succeeds either way; hypothesis 2 is EXPECTED to find one).

## Expected next artifact

`results_composition.md` + `data/beta_combo.csv` + `scripts/{score_combo,
search_v4}.py` + strategy map v4 + two e2e validations. Paper (79) gains its
strongest section: "levers compose predictably; the selector assembles them."
