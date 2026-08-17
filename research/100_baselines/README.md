# Phase 100 — reproducing external baselines, precisely

**Source phases:** 98 (`research/98_selector_demo/`), specifically
`results_g98_f_skipsets.md` and `results_lever_mechanics.md`; and
`research/79_paper/duplicate_check.md` for the related-work map.

**Status (2026-08-16):** design. Nothing measured, nothing registered.

---

## Why this phase exists

Phase 98 concluded that knapsack-style layer selection improves the *set* but
does not unlock the skip lever: with measured per-layer retention, skip-8
still beat no-speculation in only 2 of 6 regimes, and skipping 8 of 36 layers
was WORSE than not skipping at all in 3 of 5 clean regimes.

**That conclusion is about whole-layer skipping, and KnapSpec does not skip
whole layers.** Reading the paper (arXiv 2602.20217, ICML 2026):

> KnapSpec selects **Attention and MLP blocks independently**. Each is a
> separate knapsack item with its own latency weight,
> `n_Attn(S)·w_Attn + n_MLP(S)·w_MLP = k`.

Their Figure 3 shows the payoff: **skipped Attention blocks rise with context
length while MLP selections stay roughly constant.** Attention cost scales
with context, MLP does not, and their optimiser exploits that asymmetry.

Our engine cannot express that configuration —
`VLLM_SELF_SPEC_DRAFT_SKIP_LAYERS` removes attention and MLP together. So
Phase 98 did not run their method on a weaker lever; **it ran a different
method.** Any head-to-head claim resting on it is not yet earned.

## Objective

Reproduce the external baselines faithfully enough that a comparison against
our composed selector is defensible — including the ones that would beat us.

## The gaps to close, ranked

| # | gap | ours | theirs | kind |
| --- | --- | --- | --- | --- |
| 1 | **granularity** | whole transformer layer | Attention and MLP independent | **engine work** |
| 2 | **context** | ~500 tok at R1/R6/R8 | ~16K input (GovReport), to 32K generation | config |
| 3 | **depth** | fixed K=4 | D=10, early exit at `tau_conf=0.7` | config + policy |
| 4 | **scale** | Qwen3-8B | headline on Llama3.1-70B | hardware-bound |
| 5 | value proxy | measured LOO retention | cosine similarity of hidden states | ours is stronger |

(1) and (2) compound, and are the reason to expect a different answer. Our own
Nsight account measures **attention at 10.9% of the draft at batch 1, short
context**, rising to 18.9% at batch 32 — so preferentially skipping attention
buys almost nothing in the regimes Phase 98 tested, and a great deal at 16K.
Gap (5) runs in our favour: Phase 90 falsified the cosine class as an
acceptance proxy (CLaSp-cosine rho 0.37), so G98-F fed their solver BETTER
inputs than they use and it still did not unlock the lever — a real datum,
but only for whole-layer granularity.

## Baselines in scope

Ordered by how much they threaten our claims, not by ease.

1. **KnapSpec** (2602.20217) — the skip lever's strongest form. Published
   1.47x on Llama3.1-70B/GovReport against DEL 0.87x, SWIFT 1.33x,
   CLaSp 1.22x.
2. **SWIFT** — contiguous/middle-block skip. Our current skip arm is
   SWIFT-shaped, so this is close to already-reproduced and is the cheapest
   calibration point.
3. **CLaSp** — cosine-driven dynamic layer selection; the proxy class Phase 90
   falsified. Reproducing it tests that falsification end to end rather than
   by correlation.
4. **DEL** — depth/exit adaptation. Scores **0.87x** in their own table, i.e.
   below AR, which makes it a useful sanity anchor: a faithful reproduction
   should also land below 1.0.

## Assumptions

* Qwen3-8B dense as the primary column; Llama3.1-70B is out of reach on one
  H100 and their headline cell is therefore NOT directly reproducible here.
  Comparisons stay at 8B and say so.
* Training-free and distribution-preserving throughout, as in Phase 98.
* Decode currency and equal work (`ignore_eos`), per the phase-98 protocol.
* Every speedup reported against **stock vLLM AR**, not our OFF path, per
  `98/results_offpath_vs_stock.md` (our OFF is 3.7% slow).

## Design

### Stage A — sub-block skipping in the engine (the load-bearing item)

Separate attention-skip and MLP-skip masks so the draft can drop attention at
layer *i* while keeping its MLP. Without this we are not running KnapSpec, and
no comparison is meaningful. Verification: a mask that drops all attention and
no MLP must measure a cost cut close to the attention share the Nsight account
already reports for that regime.

### Stage B — a long-context regime

16K input matching GovReport's average, where the attention/MLP asymmetry
exists. Phase 98's R5/R5cot are 14K and the closest existing cells.

### Stage C — the baselines, each on its own terms

Each reproduced with ITS OWN selection rule and ITS OWN depth policy, not
forced onto our lattice. A baseline bent to our harness is not a baseline.

## Decision criteria

To be registered with a digest barrier before the first scored boot.

* **B1** the sub-block mask is faithful: attention-only skipping cuts draft
  cost within 20% of the attention share Nsight measures for that regime.
* **B2** KnapSpec reproduced at 16K beats our whole-layer skip arm at matched
  cost budget. *If this fails, Phase 98's skip conclusion generalises; if it
  passes, that conclusion is scoped to whole-layer granularity and must be
  restated.*
* **B3** DEL lands below 1.0x, as in their own table — the anchor that the
  reproductions are faithful rather than uniformly flattering to us.
* **B4** our composed selector against the best reproduced baseline, at 8B,
  reported against stock AR.

## Risks

* **B2 passing is the expensive outcome**, because it means Phase 98's skip
  finding is narrower than written and several documents need restating.
  Registering it before measuring is the point.
* **Reproduction fidelity is the whole risk.** A baseline that underperforms
  because we implemented it badly is worse than no comparison. B3 exists
  precisely to catch flattering-to-us errors, and each baseline's own numbers
  on its own reported cell should be checked where the hardware allows.
* **The 70B headline is not reproducible here.** Their 1.47x is a 70B cell; at
  8B the draft/verify economics differ (Phase 98 measured a draft forward
  costing MORE than a verify forward at batch 1). Comparisons must not imply
  we reproduced their headline.

## Open item inherited

`paper/c1.md` cites "KnapSpec's published 1.43x"; the paper's abstract says
**up to 1.47x** and its Llama3.1-70B/GovReport table reads 1.47x. The 1.43
traces to `research/89_dram_lever_swap/results_skip.md`. Reconcile before any
head-to-head claim ships — a comparison resting on the wrong published number
is the first thing a reviewer checks.

## Expected next artifact

`design_subblock_skip.md` (the engine change and its verification), then
`results_b1_mask_fidelity.md`.
