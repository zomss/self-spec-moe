# Phase 88 — canonical regime × dataset benchmark (fix the eval data)

Source: user directive 2026-07-19 ("Fix the evaluation dataset. Refer
the previous work and set the dataset appropriate for each regime"),
building on 82 (runtime switching, fixed-skip record) and 86 (KnapSpec
h2h). This phase is step 0 of the beat-all-baselines program:

0. THIS PHASE: canonical datasets per regime, grounded in prior work.
1. Close AR gaps: framework must find >=1 setting beating AR per
   regime (dig the losing regimes: short-ctx burst, summarization/doc,
   MLA, MoE).
2. Close KnapSpec gaps (32B prose parity cell).
3. Runtime lever adaptation w/ DRAM-cached quantized weights
   (EfficientRollout-style) — RL post-training; AFTER 1-2.

## Prior-work dataset provenance

| work | eval data | notes |
|---|---|---|
| KnapSpec (2602.20217) | MATH-class reasoning, b1, greedy + T=0.7 | our h2h anchor (86) |
| EfficientRollout (75) | math prompts, T=1.0, RL rollout shape | step-3 target regime |
| Spec-Bench (standard SD suite) | MT-Bench, WMT14 de-en, CNN/DM summarization, NQ-open QA, GSM8K, RAG | the community regime axes |
| LayerSkip / self-spec line | CNN/DM, XSum, HumanEval | low-accept summarization + code |
| our 82 traces | AIME CoT, C4 14k RAG, C4 continuation "docs" | docs = ARTIFICIAL -> replace |

## The regime × dataset matrix (the fix)

Regimes are map cells (batch, ctx, content). Each gets a REAL dataset
with prior-work precedent; the artificial C4-continuation regime is
replaced by CNN/DM summarization (the canonical low-accept task).

| id | regime (b, ctx) | dataset | precedent | expectation |
|---|---|---|---|---|
| R1 | interactive math CoT (b1, 2k) | GSM8K + AIME | KnapSpec, Spec-Bench | spec-favorable |
| R2 | interactive conversation (b1, short) | MT-Bench (80 q) | Spec-Bench | mid |
| R3 | code generation (b1/b8, short) | HumanEval | EAGLE-line | high-accept |
| R4 | summarization (b8/b16, 8-14k doc) | CNN/DM long docs (+gov-style long) | Spec-Bench, LayerSkip | LOW-accept: the honest hard regime |
| R5 | RAG QA/reasoning (b8/b16, 14k) | NQ-open + retrieved ctx; C4+AIME RAG kept as long-CoT variant | Spec-Bench RAG | deep-cell win regime |
| R6 | high-batch burst (b32, 2k) | GSM8K, 256-tok answers | our map's adversarial cell | spec-losing (compute-bound) |
| R7 | translation (b1/b8, short) | WMT14 de-en | Spec-Bench | mid |
| R8 | RL rollout (b16 drain, T=1.0) | MATH prompts | EfficientRollout | step-3 target |

## Plan

- E0: loader module `scripts/regime_datasets.py` — one entry point
  `load_regime(id, tok, n, ctx_target)` -> prompts + SamplingParams
  spec; downloads pinned to /data HF cache; smoke test all regimes.
- E1: AR baselines per regime cell (the denominators everything else
  is judged against), serving driver, fixed stack.
- E2: current best lever per regime from the map (W4A8-Humming/W4+win
  at 8B) -> the gap table: which regimes still lose to AR on REAL data.
  This table is the work order for step 1.
- Discipline: 82's trace machinery reused (e2_demo phase pattern);
  8-iter medians where feasible; all arms same prompts.

## Constraints

GPUs 0,1,6,7 (this box); caches on /data; .venv/bin/python.

## Artifacts

results_regimes.md; scripts/regime_datasets.py; data/ar_baselines.json,
gap_table.md.
