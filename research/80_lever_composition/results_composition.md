# Phase 80 results

## E0 — the β-composition law: PRODUCT holds, with two STRUCTURED exceptions

48 combo cells (22 combos × 2-3 ctx × 3 models, 12 prompts × ~96 pos, paired
against 77's singles on the same refs). `data/beta_combo.csv`.

**Law**: β_combo ≈ Π β_i with median |deviation| **0.013 / 0.010 / 0.009**
(dense / moe / mla); 40/48 cells within ±0.03. The E2 search may default to
the product law and use measured pairs where available.

**Exception 1 — CONSTRUCTIVE: window rescues KV-quant on dense**
(+0.13…+0.26 over product; β 0.84-0.87 vs kvq-alone 0.60-0.75, triple
win+int4+kvq 0.81-0.85 vs product 0.55). Mechanism: the window slice stops
reading the fp8-damaged K outliers (77's root cause) — a lever that cuts a
term also cuts a co-lever's ERROR EXPOSURE on that term. Strategy caveat:
cost-moot under window (528 quantized tokens save ~nothing) — relevant to
residency, and to the composition THEORY (errors are not independent when
levers share a surface).

**Exception 2 — DESTRUCTIVE, MLA-only, long-ctx-only**: skip125 composed
with any context-affecting lever collapses at 32k (skip×lr50 −0.114,
skip×win −0.100, skip×q_fp8×lr50 −0.124; all ≤0.036 at 2k; dense/moe
win×skip track product +0.01-0.02 — the smoke's "dense destructive" signal
was 2-prompt noise). Hypothesis 2 confirmed ARCHITECTURE-CONDITIONALLY —
the composition analogue of 77's portability verdict, again isolating MLA.

**MoE composes mildly ABOVE product everywhere** (+0.005…+0.027 —
correlated hard-tokens push toward min()-like behavior). The E3-candidate
triple — **win512+lr50+q_fp8, the project's original comm-free draft,
assembled — holds β = 0.816-0.829 at every context.**

Hypothesis scorecard: H1 (product for independent-error pairs) CONFIRMED;
H2 (win×skip destructive) architecture-conditional (MLA only); H3
(composition portability tracks 77) CONFIRMED in an unexpected direction —
MLA is the MOST skip-tolerant single-lever arch AND the most destructive
composer at long ctx.

## E1 — combo cost: gate FAIL (15.6%) with a clean decomposition; measured combos are GOLD

`data/e1_combo_predictions.json` (registered) vs measured (runner combo arms):

| cell | R̂ | R measured | err | reading |
|---|---|---|---|---|
| d_kvqwin b8/16k, b32/16k | 0.799 / 0.521 | 0.772 / 0.548 | **+3.5% / −4.9%** | **falsification cell PASSES**: the model correctly priced kvq+win as cost-moot (≈ window alone) — matching E0's accept-only rescue |
| d_w4win b8/16k, b32/16k | 0.505 / 0.353 | **0.435 / 0.311** | +14-16% | the dense combo bet CONFIRMED and exceeded — measured R 0.311; composed with β 0.917 → ~1.76× roofline at b32/16k |
| m_winlr / m_winlrq b32/32k | 0.740 / 0.696 | **0.357 / 0.353** | ~+100% | model refinement found: for SWA arms the h (KV-management) term should scale with ATTENDED length, not allocated ctx (FA3 sliding window skips out-of-window block-table work). winlr ≈ win alone at NVLink — the comm term is too small here; the triple's value stays parked at the comm-bound fabric |
| ds_skip125q b8/16k, b32/32k | 0.978 / 0.892 | 1.003 / 1.304 | −2.5% / −32% | MLA quant-combo cost DOESN'T deliver (fp8-Marlin κ on MLA at scale — V3's per-arch κ lesson again); MLA's OFF region survives its best challenger |

**Also found: the over-capacity detector false-positives on fresh big-batch
arms** — admission queueing during the COLD warm-pass prefill trips the
"Waiting:" grep (d_w4win/d_kvqwin b32/16k flagged with visibly clean, tight
decode TPOTs). Detector should only inspect the measured-run segments.
Runner-backlog item; the two cells were kept by manual inspection.

**Consequences for E2 (the search):**
1. Use MEASURED combo R where available; the term-edit R̂ stays the default
   for unmeasured combos on dense (validated ±16%) but is conservative for
   windowed MoE combos pending the h-term refinement.
2. Strategy insight already visible: at NVLink, window-ALONE beats the
   comm-free triple (β 0.98 vs 0.83 at ~equal R) — the triple's regime is
   the deferred comm-bound fabric, exactly as the original thesis said.
3. The dense map gains its first composed winner: **w4win at b32/16k,
   measured R 0.311, composed ~1.76× roofline** (vs 1.56× single-lever v3).

## E3 — e2e: composition CONFIRMED at b8 (91% delivery); the b32 miss exposes delivery(γ, R)

Real spec-vs-nospec tok/s, composed draft = W4 ckpt (draft_model, Marlin) +
window-KV over shared target KV — both levers running together e2e for the
first time (`scripts/e3_combo.sh`, `logs/e3c_*`):

| cell | registered | measured | accept | delivery |
|---|---|---|---|---|
| dense b8/16k K=4 | 1.55× | **1.41×** (1327/943 tok/s) | 4.39/5 | **91% — CONFIRMED** |
| dense b32/16k K=6 | 1.91× | 1.48× (3266/2219) | 5.70/7 | 77% |
| dense b32/16k K=4 | — | 1.48× (3290/2219) | 4.30/5 | (flat vs K=6) |

- **The composition LAW passed e2e**: implied β ≈ 0.90 at both cells vs the
  measured-combo 0.917 — the accept side of the composed draft delivered.
- **The b32 miss is a NEW systems law, not a composition failure**:
  delivery is a function of (γ, R), not γ alone. The composed lever cut
  draft BYTES to R=0.31, but each PIECEWISE draft step still pays the fixed
  launch floor (P72), which now dominates a much-cheaper step — so delivery
  decays exactly when composition succeeds (K=4 ≈ K=6 measured is the
  floor's signature). Composition shifts the draft bottleneck from bytes to
  LAUNCH — P74's "CUDA-graph the chain" conclusion resurfaces as the
  binding constraint at the composed frontier.
- Consequence: measured composed best at b32/16k (1.48×) lands BELOW the
  measured single-window 1.54× (76-E3) despite far lower R. On the CURRENT
  harness, the map's composed cells must be read through delivery(γ, R);
  the composed configs' full value is gated on a CUDA-graphed (or otherwise
  floor-free) draft chain — the next systems lever, now with a measured
  payoff attached (~1.9× available at b32/16k).
