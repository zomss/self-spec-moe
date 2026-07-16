# Phase 84 — pool completeness: the taxonomy and the missing gates

Source phases: 79 (menu-extension gate: pruning dead-by-domination, layer
sets alive), 83 (profiled forms; map v5), reviewer pass 2026-07-17: the
defensible form of "we explored all possible combinations" is completeness
BY TAXONOMY — every (draft-cost-term x training-free reduction op) cell is
in-map, beta-gated dead, measured-excluded with reason, or explicitly
scoped out. Six cells are silently empty; two are embarrassing (n-gram
drafting ships in vLLM; draft top-C ships in OUR OWN FORK).

## The taxonomy (status at phase open)

| cost term | reduction op | status |
|---|---|---|
| weight read | int quant (RTN/GPTQ) | IN MAP (calibrated E3-83) |
| weight read | fp8 quant | IN MAP |
| weight read | pruning (2:4/unstructured) | GATED DEAD (79: dominated by int4) |
| weight read | low-rank factorization (SVD-r) | **UNGATED -> E1 (svdr)** |
| weight read | expert restriction (contig/freq) | IN MAP (83: profiled) |
| layer compute | whole-layer skip (contig/sets) | IN MAP (83: profiled) |
| layer compute | sub-layer skip (MLP-only / attn-only) | **UNGATED -> E1 (mlpskip/attnskip)** |
| MoE routed compute | top-C prune (draft_topc, IN THE FORK) | **UNGATED -> E1 (topc)** |
| KV read | fixed window+sinks | IN MAP |
| KV read | draft-only KV quant | MEASURED-EXCLUDED (pool unbuilt; V-only rule stated) |
| KV read | dynamic token selection (Quest/PillarAttn) | SCOPED OUT (cited as upper bound on window) — candidate for a later gate |
| KV read | KV-head merging | **UNGATED -> E1 (deferred if svdr-class domination argument lands)** |
| lm_head | vocab restriction (FR-Spec-class) | **UNGATED -> E1 (vres)** |
| whole draft model | n-gram / prompt-lookup (NO model) | **UNGATED -> E1 (ngram)** — R=0, floor-free by construction |
| draft policy | tree/multi-draft | SCOPED OUT (changes the tau formula, not the lever pool) |
| draft policy | gamma | IN SELECTOR |

## Plan

- **E1 — the six gates** (beta on 77's paired refs, ~30-60 min each):
  ngram (CPU-only: coverage x accuracy on cached refs; note R=0 AND
  floor-free — prices by tau alone), topc C=4/2 (moe), mlpskip/attnskip
  (dense, budget-matched to the layer-set frontier), vres top-16k/32k
  (dense, lm_head hook), svdr at 50%-bytes rank (dense, destructive).
- **E2 — verdicts into the taxonomy + map v6 pool**: alive levers get cost
  specs and enter the search; dead levers get domination arguments.
- Gate: every taxonomy cell non-empty, with measured or argued status.

## Constraints

- GPUs 6-7 (user-directed 2026-07-17); caches on /data.
- Paired refs (77 ondist) for every gate; ngram needs NO GPU.
- Beta method unchanged (1152 positions); arms destructive-last.

## Expected artifact

`results_pool.md`: gate table + updated taxonomy; feeds the paper's 5.4
(pool-completeness claim upgraded from menu-extension to taxonomy) and
map v6.
