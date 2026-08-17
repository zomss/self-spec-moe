# Final evaluation grid: long-in / long-out / long-both, natural EOS

Record of 2026-08-16. Design, nothing registered. Companion to
`dataset_map.md`: the P1–P5 blocks there are *fidelity* cells run under
each baseline's own protocol; **this grid is the phase's scoring protocol —
ours** — on which every method (ours and reproduced baselines) is compared
under one rule set.

## Decisions taken (2026-08-16)

1. **Coverage by length quadrant**: long input, long output, and
   long input + long output are the target cells. A short/short control row
   is added (my addition, cheap): it anchors against the skip family's
   Spec-Bench-class literature, and it is the regime where the fail-closed
   rule should visibly decline to speculate.
2. **EOS is respected. `ignore_eos` is retired from scored evaluation.**
   Phase 98's equal-work convention remains a *measurement instrument*
   (lattice calibration, cost attribution) but no scored Phase-100 number
   uses it.
3. **Every cell sweeps batch size**, clamped by measured KV feasibility.
4. Datasets must exist in the baseline literature (`dataset_map.md`) — no
   C4 filler, no homemade packing in scored cells.

## Why retiring `ignore_eos` is nearly free — and what it buys

All methods in scope are distribution-preserving: at T=0 the verified
output is bit-identical to the target's AR output. So every arm generates
**the same tokens and stops at the same EOS** — equal work across arms is
preserved by losslessness itself, not by forcing generation on. Two
consequences:

* **Free correctness gate**: any cross-arm output divergence at T=0 fails
  the run outright. This gate is registered as part of the protocol.
* The one real cost: **output length becomes model-determined**, so cell
  cost is known only after a pilot run. Caps below are safety nets, not
  budgets; the fraction of requests hitting a cap is reported per cell and
  a cell with >10% cap-hits gets its cap raised and is re-run.

Casualty: **PG-19 is incompatible with natural EOS** (book continuation
does not emit EOS) — it stays in the fidelity block P1 (MagicDec's
protocol, capped gen) and never enters this grid.

At T=1 outputs differ across arms by sampling; the LO cell carries a
secondary T=1 arm (R8 heritage) scored over multiple content seeds, with
length distributions reported alongside rates.

## The grid

| cell | dataset | input | output (natural) | cap | batch sweep | T |
| --- | --- | --- | --- | --- | --- | --- |
| **LI** | GovReport | ~16K | ~0.3–1K (summary) | 2K | 1 / 8 / 16 | 0 |
| **LO** | AIME 2024+2025 | ~0.1–0.5K | ~5–25K (thinking CoT) | 32K | 1 / 8 / 16 | 0 primary, 1 secondary |
| **LIO** | BookSum (chapter, length-filtered ≥8K) | ~8–16K | ~1–2K (long summary) | 4K | 1 / 8 / 16 | 0 |
| **SS** | GSM8K, "answer concisely" | ~0.3K | ~0.1–0.3K | 1K | 1 / 8 / 32 / 64 | 0 |

Dataset notes:

* **LI = KnapSpec's headline data** (GovReport, ~16K average input) and
  Stage B in one cell. Summaries terminate naturally.
* **LO = the KnapSpec direct-overlap cell**: their reasoning column runs
  Qwen3-8B on AIME24/25 at a 32K generation budget. **Vintage switch
  required**: our R1/R8 loaders use `AIME_1983_2024`; this cell uses AIME
  2024 + 2025 problem sets to match theirs (exact HF dataset ids to be
  pinned at loader-build time).
* **LIO** is the quadrant no baseline paper isolates cleanly; BookSum is
  the in-literature choice (KnapSpec uses it) with chapter-level entries
  length-filtered by the Qwen3 tokenizer. Alternatives considered:
  Multi-LexSum (~90K inputs — exceeds practical context here),
  R4-style multi-article packing (homemade — excluded by decision 4).
* **SS** inherits R6's shape (GSM8K, concise) and extends the sweep to
  b64. CNN/DM single-article stays in fidelity block P4, not here, to
  avoid duplication.

## Batch feasibility (computed, to be verified by preflight)

Qwen3-8B KV = 144 KB/token BF16 (36 layers x 8 KV heads x 128 dim x 2).
H100 80GB minus ~16GB weights minus overhead => **~60GB KV budget**.

| cell (in+out) | b8 | b16 | b32 |
| --- | --- | --- | --- |
| LI ~16.6K | 18 GB | 37 GB | **73 GB — infeasible** |
| LO ~15K | 17 GB | 33 GB | **66 GB — infeasible** |
| LIO ~13.5K | 15 GB | 30 GB | 59 GB — marginal, excluded |
| SS ~0.6K | 0.6 GB | — | 2.4 GB (b64: 4.9 GB) |

So the long cells stop at b16 — a physical constraint of 8B-on-one-H100,
recorded rather than hidden. Draft-side KV (window/skip arms) adds on top
of this; `99_kv_pressure/scripts/preflight_w99_kv_budget.py` is the
instrument to verify per-arm headroom before the barrier is registered.
Numbers above use expected natural lengths; the pilot run replaces them
with measured length distributions.

## Metric

* Fixed request set per cell, **n = 4 x max batch** (long cells n=64,
  SS n=256), submitted at once; continuous batching drains the set.
* Primary: **wall-clock time to drain the set**; speedup = time ratio
  against stock vLLM AR (per `98/results_offpath_vs_stock.md`, never our
  OFF path). Identical outputs at T=0 make the ratio exact equal-work.
* Secondary: tokens/s over the drain, per-step telemetry (acceptance,
  draft-chain cost) via the existing instruments for diagnosis only.
* Ragged tails are part of the measurement, not an artifact to remove —
  with EOS respected, drain behaviour under mixed completion lengths is
  precisely the serving reality the batch sweep exists to capture.

## Arms (every cell x every batch)

1. stock vLLM AR (denominator)
2. our OFF path (instrument-cost audit, carried from X32)
3. unmodified self-draft (the 0.89–0.94x floor, made visible)
4. best single lever per family: w4a16; w1024 window; knapsack skip
5. MagicDec fixed-budget row (KV-family baseline, native in our engine)
6. reproduced skip baselines as they land (Self-SD / SWIFT / KnapSpec via
   Stage A; CLaSp after dynamic-mask work)
7. composed selector with fail-closed rule — the phase's subject

## Registration checklist (before first scored boot)

- [ ] pilot run -> measured length distributions -> final n, caps, batch clamps
- [ ] HF dataset ids + revisions pinned (GovReport, BookSum, AIME24/25, GSM8K)
- [ ] T=0 cross-arm output-identity gate wired as a hard failure
- [ ] cap-hit fraction reporting + >10% re-run rule
- [ ] dual-protocol fork registered: this grid (ours) vs P1–P5 (theirs)
- [ ] host-load `measurement_verdict` wired inline (G98-F lesson, not post-hoc)
- [ ] digest barrier committed, then boots
