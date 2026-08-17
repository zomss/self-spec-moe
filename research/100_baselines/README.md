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

The survey (`baseline_survey.md`) sharpened this: sub-block granularity is
**the family default since Self-SD (2309.08168, ACL 2024)**, whose search
space is already per-sublayer (2L binary variables under Bayesian
optimization). Our whole-layer skip diverges from the original 2023 method,
not merely from KnapSpec — and Stage A therefore unlocks Self-SD and
KnapSpec at once.

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

Full field, per-method details, and reproduction-cost tiers:
**`baseline_survey.md`** (record of 2026-08-16).

**Scope decision (2026-08-16):** the main baseline set is the three families
that compete lever-for-lever with our search space — **layer-skip,
KV-sparsity draft, quantized draft**. Model-free drafters (PLD/ngram,
suffix) are drafter *replacements*, not lever compositions, and are
**deferred** — cite-level for now, revisitable at one vLLM flag each.

Threat order — by how much they threaten our claims, not by ease:

1. **MagicDec** (2408.11049, ICLR 2025) — StreamingLLM sinks+window
   self-draft with a fixed KV budget. **At the lever level it is our window
   lever**; the reviewer question "isn't your window lever just MagicDec?"
   must be answered by a measured row, not argument. Runnable in our engine
   today as a fixed cell of the window lattice under their protocol.
2. **KnapSpec** (2602.20217, ICML 2026) — the skip lever's strongest form.
   Published 1.47x on Llama3.1-70B/GovReport against DEL 0.87x, SWIFT 1.33x,
   CLaSp 1.22x. Needs Stage A.
3. **Self-SD / Draft&Verify** (2309.08168, ACL 2024) — the family origin,
   sub-block granularity, offline Bayesian optimization. Same Stage A
   machinery as KnapSpec; a KnapSpec reproduction that cannot also
   reproduce Self-SD is suspect.
4. **SWIFT** (2410.06916, ICLR 2025) — closest shape to our current skip
   arm; cheapest calibration point. Faithful version adds its online
   re-optimization interval.
5. **CLaSp** (2505.24196, ACL 2025) — cosine-driven dynamic layer selection;
   the proxy class Phase 90 falsified. Reproducing it tests that
   falsification end to end rather than by correlation.
6. **Quantized-draft family** — our w4a16 arm is itself this family's
   representative. QuantSpec (2502.10424, ICML 2025) is the closest
   published composition (4-bit weights + quantized KV draft) but needs
   kernels we do not have — cite-level unless promoted; QSpec stays
   border-excluded (its verify path is quantized, so it is not
   distribution-preserving against an FP16 target).

**DEL is excluded.** The survey established it requires LayerSkip-trained
checkpoints (its own paper: 2.16–2.62x *on those checkpoints*); KnapSpec's
0.87x is early exit on a stock model. It cannot be run under our
training-free constraint and cannot anchor fidelity (see B3').

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
* **B3'** directional fidelity anchors (replaces the original B3, which
  rested on DEL and died with its exclusion — `baseline_survey.md`):
  (a) MagicDec-style fixed budget wins at long-context/batched cells and
  loses at short-context batch-1, per their bottleneck analysis and our own
  measured activation threshold; (b) each reproduced skip method lands
  within, or *explainably* below, its own paper's reported band, where
  "explainably" means attributed to a measured mechanism such as the 8B
  draft-loses-8–11% floor. (An earlier PLD clause was withdrawn with the
  family-D scope decision.)
* **B4** our composed selector against the best reproduced baseline, at 8B,
  reported against stock AR.

## Risks

* **B2 passing is the expensive outcome**, because it means Phase 98's skip
  finding is narrower than written and several documents need restating.
  Registering it before measuring is the point.
* **Reproduction fidelity is the whole risk.** A baseline that underperforms
  because we implemented it badly is worse than no comparison. B3' exists
  precisely to catch flattering-to-us errors, and each baseline's own numbers
  on its own reported cell should be checked where the hardware allows. The
  DEL discrepancy (0.87x in KnapSpec's table vs 2.16–2.62x in DEL's own
  paper — a ~2.7x disagreement between two published venues) is the measure
  of how badly this can go when a reproduction silently changes the setup.
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

`baseline_survey.md` — **done, 2026-08-16.** Next:
`design_subblock_skip.md` (the engine change and its verification; before
freezing it, resolve the survey's open item on SWIFT/CLaSp/ConfLayers
granularity so one mask implementation serves the whole family), then
`results_b1_mask_fidelity.md`.
