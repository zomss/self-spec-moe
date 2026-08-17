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
| **LO** | AIME 2024+2025 | ~0.1–0.5K | ~5–25K (thinking CoT) | 32K | 1 / 8 / 16 / 32 | 0 primary, 1 secondary |
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

## Batch feasibility — MEASURED (pilot of 2026-08-16)

Pilot: `scripts/run_w100_pilot.py` on h104 lane GPU 7, stock decode,
seed 0; record `data/pilot/pilot_lengths.json`; analysis
`scripts/analyze_w100_pilot.py` -> `data/pilot/pilot_feasibility.json`.

**Measured natural lengths** (output tokens, EOS respected):

| cell | n | median | p95 | max | cap hits | cell wall |
| --- | --- | --- | --- | --- | --- | --- |
| LI | 16 | 771 | 999 | 999 | 0 | 29 s |
| LO (T=0) | 16 | 9,567 | 32,768 | 32,768 | **2/16** | 356 s |
| LO (T=1) | 8 | 14,334 | 31,936 | 31,936 | 0 | 277 s |
| LIO | 16 | 1,257 | 1,459 | 1,459 | 0 | 40 s |
| SS | 32 | 153 | 350 | 444 | 0 | 5 s |

**Feasibility is drain-aware, and that changes the clamps.** The design
draft used a static bound (p95 length x batch held simultaneously),
which contradicts the protocol itself: with EOS respected, requests
finish raggedly and FREE their KV, so the binding quantity is peak
concurrent KV over the drain, simulated from measured lengths
(`analyze_w100_pilot.py`). Against a ~55.6 GB pool:

| cell | b8 | b16 | b32 | b64 | clamp |
| --- | --- | --- | --- | --- | --- |
| LI | 16 GB | 34 GB | **68 GB** | — | **{1, 8, 16}** |
| LO | 10 GB | 13 GB | 26 GB | **53 GB — marginal, excluded** | **{1, 8, 16, 32}** |
| LIO | 14 GB | 33 GB | **66 GB** | — | **{1, 8, 16}** |
| SS | 0.1 GB | 0.4 GB | 0.6 GB | 1.3 GB | **{1, 8, 32, 64}** |

The static model was right where prompts dominate (LI/LIO: all prompts
co-resident from admission) and wrong where outputs dominate (**LO
extends to b32** — tiny prompts, ragged drain). Draft-side KV
(window/skip arms) adds on top; verify per-arm headroom with
`99_kv_pressure/scripts/preflight_w99_kv_budget.py` before registration.

### LO cap policy (pilot amendment)

LO's 2/16 T=0 cap-hits exceed the 10% raise-and-rerun rule, but the cap
is NOT raised, for three registered reasons: (1) **KnapSpec parity** —
32K is their own AIME generation budget, so matching it is the point of
the cell; (2) the cap sits near Qwen3-8B's native 40,960 anyway; (3) at
T=0 cap-hits are **arm-invariant** — losslessness means every arm
generates the same greedy stream and caps at the same step, so capped
requests contribute exactly equal work and no arm comparison is biased.
LO is therefore exempted from the 10% rule; cap-hit fraction is reported
with every LO number. The T=1 arm finished 8/8 under the cap (median
14.3K), consistent with the T=0 cap-hits being greedy-thinking
repetition loops rather than genuine 32K reasoning — to be confirmed by
inspecting the capped outputs at analysis time, not blocking.

### Per-cell n (pilot amendment)

The draft's flat "n = 4 x max batch" is unaffordable at LO (128 requests
x ~12K tokens at b1 is ~8 h per arm). Replaced with per-(cell, batch)
**n = max(16, 2b)**, with nested prompt sets (the b=32 set contains the
b=16 set, etc.) so every arm sees identical prompts and comparisons are
paired; T=0 determinism makes paired small-n exact rather than noisy.

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

- [x] pilot run -> measured length distributions -> final n, caps, batch
      clamps (2026-08-16, this document's measured sections)
- [x] loaders with pinned revisions (`scripts/w100_eval_datasets.py`)
- [x] HF dataset ids + revisions pinned AND prompts frozen to
      `data/registration/w100_prompts.jsonl.gz` + manifest
      (`scripts/generate_w100_prompt_manifest.py`)
- [x] T=0 cross-arm output-identity gate wired as a hard failure
      (`w100_protocol.identity_gate`)
- [x] cap-hit fraction reporting + >10% re-run rule with the registered
      LO exemption (`w100_protocol.cap_rule`)
- [x] dual-protocol fork registered (`w100_prereg.md`)
- [x] host-load gates inline (G98-F lesson): `w100_protocol.boot_gate`
      + `measurement_gate` (phase-98 limbs re-fed from the SS batch
      sweep; spread limb to recalibrate on first clean boots)
- [x] digest barrier: `data/registration/w100_barrier.json`
      (`scripts/make_w100_barrier.py`), committed before any scored boot
