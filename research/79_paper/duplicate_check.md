# Duplicate-work check (fail-closed gate) — 2026-07-13

Claim checked: **a measured cost×accept selector for training-free
self-speculative draft levers across (batch, context, architecture), with an
OFF region, e2e validation on five cells, and one-anchor transfer to an
unswept architecture.**

## Verdict: NOT duplicated — proceed, with three required sharpenings

Live search (2026-07): the field is crowded with SINGLE-lever methods and
with selectors over OTHER axes (depth, drafter identity, overlap), but no
work does measured multi-lever × multi-architecture selection with
acceptance-portability findings and validated transfer.

## Nearest neighbors and the delta

| prior art | covers | our delta |
|---|---|---|
| **MagicDec** (2408.11049) | THE seed framing: bottleneck-aware analysis over batch×seq-len, dense models, KV-sparse self-draft; "select drafting strategy" language | extends to MoE/EP + the architecture axis; a measured multi-lever portfolio (quant, skip, local-route, kvq — not just KV-sparse); measured β portability (only fp8-weight portable); OFF region; one-anchor transfer; 5-cell e2e validation |
| **SparseSpec** (2512.01278; Zhao, Tang, ..., Han, Stoica) | our WINDOW lever as a full paper: sparse-attn self-spec (PillarAttn — verification-informed token selection, better than fixed sinks+window) + serving co-design; dense reasoning models | single lever, no selection, no MoE/EP, no β-across-architectures, no cost model. MUST CITE; position our fixed-window arm as a baseline instance of their lever — our contribution is the lever's PLACE in the measured map (incl. where it LOSES: MLA, short ctx), not the lever |
| **Not-a-Bandit** (2510.20064) | ONLINE no-regret selection among trained DRAFT MODELS, per query | orthogonal selection axis: chooses among deployed drafters given traffic; ours chooses among SELF-spec levers from measured physics, per architecture, incl. OFF; composable (their online layer atop our offline map) |
| **Learning to Draft** (2603.01639, ICLR'26) / SmartSpec-class / PACER / DEL (2504.05598) | adaptivity over DEPTH/length (RL, goodput, exit layer) | γ/depth adaptation inside one lever ≠ lever selection; our γ* falls out of the same composition; DEL is the skip lever's depth-adaptive cousin — cite beside our skip β-curve |
| **SS-MoE** (WebConf'26), **MoE-Spec** (2602.16052), **SP-MoE** (2510.10302), utility-driven MoE-spec (2506.20675), adaptive MoE verification (2605.00342) | MoE-specific single mechanisms (expert-subset draft, verify-side expert budgeting, prefetch) | none selects among levers or maps regimes; several are complementary (MoE-Spec's verify budgeting composes with any draft lever) |
| **Speculative Speculative Decoding** (ICLR'26, Kumar/Dao/May) | overlaps drafting with verification | orthogonal (cite with our P32 overlap analysis) |

## Required sharpenings (from the check)

1. **Position against MagicDec explicitly as the seed**: our paper is "MagicDec's
   regime question, answered with a measured multi-lever, multi-architecture
   selector" — never imply the bottleneck framing is ours.
2. **Cite SparseSpec prominently for the window lever** (their PillarAttn
   likely dominates our fixed sinks+window; our map's window numbers are a
   LOWER bound on that lever) — and note our unique window findings they lack:
   window-size β-insensitivity on GQA, its collapse on MLA, the MLA cost+β
   double-weakness.
3. **Disambiguate "selector"**: three selection axes exist in 2026 — depth
   (SmartSpec/LTD), drafter identity (Not-a-Bandit), and OURS: lever ×
   architecture from measured cost×accept physics with OFF. State the axis.

## Sources

- MagicDec: https://arxiv.org/pdf/2408.11049
- SparseSpec: https://arxiv.org/pdf/2512.01278
- Not-a-Bandit: https://arxiv.org/abs/2510.20064
- Learning to Draft: https://arxiv.org/html/2603.01639
- Speculative Speculative Decoding: https://openreview.net/forum?id=aL1Wnml9Ef
- DEL: https://arxiv.org/pdf/2504.05598
- SS-MoE: https://dl.acm.org/doi/10.1145/3774904.3792218
- MoE-Spec: https://arxiv.org/abs/2602.16052
- SP-MoE: https://arxiv.org/html/2510.10302v1
- Utility-driven MoE spec: https://arxiv.org/abs/2506.20675
- Adaptive MoE verification: https://arxiv.org/pdf/2605.00342
- Rethinking high-throughput spec (unread, login-walled — check at camera-ready):
  https://openreview.net/forum?id=59OJOgKLzN

## Addendum 2026-07-13: KnapSpec (2602.20217)

**KnapSpec: Self-Speculative Decoding via Adaptive Layer Selection as a
Knapsack Problem** (Cha, Kim, Han, Yang, Han; github kaist-flexml-lab).
Layer SELECTION (non-contiguous) as a knapsack over offline-profiled
per-layer latency/acceptance. Dense-only, single-lever (skip family), no
OFF, no batch/ctx regime axis. NOT a duplicate — it is to our skip arm what
SparseSpec is to our window arm: the lever's strongest form. Cite in §2
beside SWIFT/DEL; position our contiguous middle-skip as a lower bound.
Its offline-profile+combinatorial-search methodology independently
validates our §7 framing (search ACROSS levers vs their search WITHIN one).

**Triggered hardening (user review)**: OFF verdicts re-derived by
exhaustive priced search over the full combo space (79/scripts/
off_hardening.py) — see §6 update; OFF now means "the argmax over the
whole combination space loses under optimistic pricing".
