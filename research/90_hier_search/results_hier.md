# Phase 90 results — hierarchical search (DRAFT)

## E1: proxy validation against the committed beta record (2026-07-23)

Scores from ONE forward pass per (model, ref-set): per-layer angular
distance (importance) + per-(layer,head) beyond-W attention mass.
Validated against 33 measured greedy-scan arms per model (86/83 logs)
and the R4 window record. GPUs 6/7, ~15 min total.

### P1a (skip proxy): REFUTED — the naive correlation was a size
### artifact

| test | 32B | 8B |
|---|---|---|
| naive Spearman (all 33 arms) | 0.965 | 0.958 |
| set-SIZE alone vs -beta | 0.982 | 0.982 |
| within-size added-layer discrimination (pooled, n=30) | **-0.109 (p=.57)** | **-0.165 (p=.38)** |

The impressive naive correlation is entirely explained by set size
(more skipped layers -> lower beta; more summed importance). Holding
size fixed, angular-distance importance has NO detectable power to
rank WHICH layer to skip — at either scale, corroborated by the
greedy-set check: proxy bottom-k picks the network's LATE layers
(8B: 31-33; 32B: 16-19) while acceptance-greedy picks EARLY layers
(8B: 3-6; 32B: 2,4,7), overlap ~0.

FINDING (paper-grade): perplexity-class importance and ACCEPTANCE
importance diverge systematically. Compression-literature scores
rank late layers most redundant; draft acceptance wants early layers
skipped. Draft construction cannot transplant pruning recipes — the
objective (agreement with the target's next token) is different from
the objective those scores were built for (output quality). This is
the measured justification for C2's "measure the right thing" thesis.

### P2 (concentration): REFUTED at both granularities

Beyond-512 attention mass on CNN/DM 8k refs is DIFFUSE: 43.3% of
(layer,head) pairs / 50% of layers carry 70% of the mass (gate was
<=25%). No small retrieval-head set exists at raw-mass granularity
at 8B. Extra twist: the layers with the MOST far mass are the early
ones (top: 3,4,5,2,1,6) — the same layers acceptance-greedy skips.
Raw far-attention mass is largely redundant attention; selecting
full-context heads by raw mass would target exactly the wrong layers.

### P1b (window band structure): SUPPORTED (directional)

- Widening 512->2048 captures only 33% of the missing mass (measured
  accept gain: 2.70->2.67 ~ zero); 67% of the missing mass sits
  beyond 2048 — article-scale. Consistent with the measured step
  function (win8192 -> accept 4.06).
- Task dependence: beyond-512 mass c4 .1145 vs cnndm .1432 (+25%),
  directionally matching the ondist/summarization accept split.

## Consequence (per the pre-registered fallback)

The one-pass-geometry proxy stage is DEAD for acceptance ranking:
stage 2 falls back to DIRECT beta screening — which the 77-harness
makes cheap (~2-5 min/config): leave-one-out at layer granularity is
an L-run evening, not a proxy's minute, but it measures the actual
objective. The hierarchical scheme survives with the corrected cost:
  stage 1 analytic cost x kernel factor (unchanged)
  stage 2 DIRECT beta LOO/greedy at the chosen granularity
          (proxy REFUTED: measure, don't score)
  stage 3 pools + switch-cost-aware bandit (unchanged)
E3 (hetero-window) redesign: per-LAYER full-context assignment
searched by direct beta LOO (36 runs), NOT by mass scores. P3 gate
unchanged (accept >= 3.6 at near-win512 cost).
