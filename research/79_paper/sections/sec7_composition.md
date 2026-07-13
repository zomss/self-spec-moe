# §7 Lever composition (draft)

> Source: Phase 80 (`results_composition.md`, `data/beta_combo.csv`,
> `e1_combo_predictions.json`, `e3c_*` logs). Numbers final. ~1.5 pages.

## 7.1 Composition as offline profiling

Levers cut different terms of §6's cost model, so composing them should
compound R; the open question is what composition does to β. We treat this
the way the pruning and mixed-precision-quantization literature treats
configuration search: profile offline, fit a cheap composition rule,
search the composed space, and validate the argmax end to end. (SWIFT
searches WITHIN one lever — skip sets; ours is search ACROSS levers, with
the single-lever maps of §4–5 as the profile.)

## 7.2 F9 — the β-composition law

Across 48 combo cells (22 combinations × contexts × three architectures,
paired against §5's singles on identical references):

  β_combo ≈ Π β_i, median |deviation| 0.013 / 0.010 / 0.009
  (dense / MoE / MLA); 40/48 cells within ±0.03.

Acceptance errors from different levers are near-independent — the search
can price unmeasured combinations by multiplication. The two exceptions are
structured, and both teach mechanism:

**Constructive (dense): window RESCUES KV-quant** (+0.13…+0.26 over
product; kvq-alone 0.60–0.75 → win+kvq 0.84–0.87). The window slice stops
reading exactly the fp8-damaged K outliers that §5-F6 identified — a lever
that cuts a term also cuts a co-lever's ERROR EXPOSURE on that term.
(Cost-moot — quantizing 528 windowed tokens saves nothing — but it breaks
error-independence in a predictable direction: shared surface ⇒ shared
error.)

**Destructive (MLA-only, long-context-only): shallow skip composed with
any context-affecting lever collapses at 32k** (skip×lr50 −0.114,
skip×window −0.100; all ≤0.036 at 2k; the same pairs track product on
dense/MoE). Composition portability tracks §5's architecture verdict —
MLA is simultaneously the most skip-tolerant single-lever architecture and
the most destructive composer.

MoE composes mildly ABOVE product everywhere (+0.005…+0.027): hard tokens
are correlated across levers, pushing toward min()-like behavior. Notably,
the comm-free triple this project began from — window+local-route+fp8,
assembled — holds β = 0.82–0.83 at every context; §7.3 shows why it still
loses at NVLink.

## 7.3 Pricing composed cost by term edits

Combo R̂ comes from applying multiple term edits to §6's fitted model;
measured combos then grade the pricing (predictions registered first):

- **The falsification cell passes**: kvq+window is priced COST-MOOT
  (≈ window alone) precisely because the model says the window already
  removed the KV bytes kvq would quantize — measured +3.5%/−4.9%.
- **The dense composed bet exceeds its registration**: W4+window at
  b32/16k measured R = 0.311 (predicted 0.353) — composed with the
  measured β 0.917, a ~1.9× roofline where the best single lever offers
  1.56×.
- **A model refinement found by its own failure**: windowed-MoE combos
  measured R ≈ 0.35 vs predicted 0.70 — the h (KV-management) term should
  scale with ATTENDED length, not allocated context, for sliding-window
  arms. Reported as the model's current edge; dense pricing is unaffected.
- **MLA's best challenger fails on cost** (fp8 κ at tiny per-expert M —
  §6's per-architecture κ lesson), so MLA's OFF region survives
  composition.
- At NVLink, window-alone beats the comm-free triple (β 0.98 vs 0.83 at
  ~equal R): the triple's regime is the comm-bound fabric, deferred (§9).

The search over the composed space (product-β default, measured pairs
where available, measured-combo R as gold) upgrades the dense long-context
map: its winner becomes the composed W4+window configuration, with
measured-β × measured-R provenance.

## 7.4 F10 — composition end to end, and the delivery(γ, R) law

Running the composed draft live (W4 checkpoint as draft_model + window-KV
over the shared target cache):

| cell | roofline | measured | delivery |
|---|---|---|---|
| dense b8/16k K=4 | 1.55× | 1.41× | **91% — composition CONFIRMED e2e** |
| dense b32/16k K=6 | 1.91× | 1.48× | 77% |
| dense b32/16k K=4 | — | 1.48× | (flat vs K=6) |

The accept side delivered at both cells (implied β ≈ 0.90 vs measured-combo
0.917). The b32 miss is not a composition failure but a systems law:
**delivery is a function of (γ, R), not γ alone.** Composition cut draft
bytes to R = 0.31, but each draft step still pays a fixed launch cost,
which now DOMINATES a much cheaper step — K=4 ≈ K=6 in the measurement is
the floor's signature (more, cheaper steps buy nothing when each step pays
the same fixed toll). Composition succeeds by shifting the draft bottleneck
from bytes to launch — so delivery decays exactly when composition works
best, and the composed winner (1.48×) lands below the single-window 1.54×
despite far lower R.

This is the cliff-hanger §8 resolves: the floor is measurable, attributable
— and removable. With it removed, this table's b32 row becomes 1.91× at
~100% delivery.

## 7.5 The cost of the search: a profiling-budget backtest

The selection step is free — pricing all 53 configurations per cell is CPU
work. The search's real cost is PROFILING, so we ask: how few GPU-minutes
buy the same decisions? We backtest by replaying a minimal protocol
against the completed record, treating MLA as the new architecture (which
is how the project actually unfolded), revealing at each budget level only
what the protocol would have measured, and scoring every cell's argmax
against the full-information map. Regret = true speedup of the chosen
config vs the true best.

| budget level | GPU-min | near-opt (≤0.02) | median regret | max regret |
|---|---|---|---|---|
| L0: mechanism priors only | 0 | 1/9 | +0.077 | +0.223 |
| L1: + one bf16 R anchor | 5 | 1/9 | +0.077 | +0.223 |
| L2: + one κ anchor + checklist β at one ctx | 26 | 3/9 | +0.057 | +0.213 |
| L3: + deviation-triggered ctx reveal | 46 | 3/9 | +0.064 | +0.213 |
| **L4: + contested-cell R + conservatism rule** | **91** | **8/9** | **+0.000** | **+0.026** |
| full profile | 305 | 9/9 | 0 | 0 |

Three lessons. (1) **Priors alone fail** (max regret 0.22): per-kernel κ
and per-architecture context behavior are not config-derivable — the maps
are load-bearing, not decoration. (2) The last mile comes from DECISION
RULES, not more data: measure R only where the top-2 gap is inside the
model's validated error (±0.09), and on such ties prefer measured-R
configs over model-priced ones — that one conservatism rule alone cut max
regret from 0.213 to 0.026. (3) The protocol's cost is roughly
grid-independent (two anchors + one β column + a handful of contested
cells), so its advantage grows with map size: here 91 vs 305 GPU-minutes
(3.4×) for one 9-cell architecture. This is the search protocol we
recommend over exhaustive profiling: exhaustive SELECTION over measured
physics, with measurement itself allocated by decision uncertainty.
