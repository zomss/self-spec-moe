# W100 preregistration — final evaluation grid

Registered 2026-08-16, before any scored boot of this phase. The barrier
(`data/registration/w100_barrier.json`) pins this document, the design
record, the loaders, the protocol module, and the frozen prompt sets by
sha256. Any change to a pinned artifact after the barrier commit is a
protocol amendment and requires a superseding registration; results
produced under a superseded barrier say so.

## Grid

Cells, caps, and batch clamps as measured and decided in
`final_eval_design.md` (pilot of 2026-08-16):

| cell | data (pinned in manifest) | cap | batches | T |
| --- | --- | --- | --- | --- |
| LI | GovReport test, band 12–20K | 2048 | 1 / 8 / 16 | 0 |
| LO | AIME 2024+2025, thinking on | 32768 | 1 / 8 / 16 / 32 | 0 |
| LO_T1 | same prompts, first 16 | 32768 | 8 | 1.0, seed 1234 |
| LIO | BookSum aggregates, band 8–20K | 4096 | 1 / 8 / 16 | 0 |
| SS | GSM8K test, "answer concisely" | 1024 | 1 / 8 / 32 / 64 | 0 |

Per-(cell, batch) request count **n = min(cell_max, max(16, 2·batch))**,
nested prefixes of the frozen prompt file, identical across arms (paired
comparisons). Cell maxima: LI 32, LO 60 (all unique AIME problems),
LIO 32, SS 128.

## Protocol rules (enforced in code: `scripts/w100_protocol.py`)

1. **EOS respected everywhere.** No `ignore_eos` in any scored number.
   Caps are safety nets; cap-hit fraction is reported per (cell, arm).
2. **Cap rule:** cap-hit fraction > 10% fails the cell and forces a
   raised cap and re-run — **except LO**, exempt for three registered
   reasons: 32K is KnapSpec's own AIME budget (parity is the cell's
   purpose); the cap sits near Qwen3-8B's native 40960; and at T=0
   cap-hits are arm-invariant, contributing exactly equal work.
3. **Identity gate (hard failure):** at T=0 every arm's output text
   sha256 must equal the same campaign's stock-AR arm, per prompt. Any
   divergence is a lossless-verify correctness alarm and voids the boot.
4. **Boot gate:** no boot on a lane with foreign processes
   (`w98_host_load.foreign_lane_processes`), fail closed.
5. **Measurement gate (inline, the G98-F lesson):** the SS batch sweep's
   mean step time must rise monotonically in batch and keep relative
   spread ≥ 0.03 (phase-98 clamp signatures, same two limbs re-fed from
   W100's own cells). Violations reject the boot at the boot, not in
   post-hoc annotation. Spread limb to be recalibrated on the first
   clean W100 boots; monotonicity is load-bearing.
6. **Prompts come from the frozen file**
   (`data/registration/w100_prompts.jsonl.gz`), never from live HF
   loads.

## Metric

Per (cell, batch, arm): submit the n registered requests at once; score
**wall-clock to drain**. Speedup = stock-AR drain time / arm drain time
on identical prompts. Denominator is **stock vLLM AR** (no
speculative_config, no self-spec env), never our OFF path
(`98/results_offpath_vs_stock.md`). Tokens/s and per-step telemetry are
reported as diagnostics, not scores.

## Arms

stock AR · OFF · unmodified self-draft · w4a16 · w1024 window ·
knapsack skip · MagicDec fixed-budget (window lattice cell under their
budget policy) · reproduced baselines as they land (Stage A onward) ·
composed selector with fail-closed rule.

## Dual-protocol fork (registered once, here)

This grid is **our** protocol. The fidelity blocks P1–P5
(`dataset_map.md`) run under **each baseline's own** protocol (their
datasets, gen budgets, sampling) and exist to place reproductions inside
their published bands (B3'). A baseline's headline row in any comparison
table must state which protocol produced it. Numbers from the two
protocols are never mixed in one column.

## Decision criteria (restating README §Decision criteria, now bound to
this grid)

* **B1** — Stage A sub-block mask fidelity: attention-only skipping cuts
  draft cost within 20% of the Nsight-measured attention share.
* **B2** — KnapSpec reproduced at long context (LI/LIO) beats our
  whole-layer skip arm at matched cost budget → Phase 98's skip
  conclusion is scoped to whole-layer granularity and must be restated.
  Registered as the expensive outcome.
* **B3'a** — MagicDec fixed-budget wins at long-context/batched cells
  and loses at short-context batch-1.
* **B3'b** — each reproduced skip baseline lands within, or explainably
  below, its own paper's band under its own protocol (P-blocks).
* **B4** — composed selector vs best reproduced baseline, this grid,
  vs stock AR.

## What is NOT registered

Runtime engine changes (Stage A), the P1–P5 fidelity runners, and any
regime beyond the five cells above. Those get their own barriers.
