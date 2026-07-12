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
