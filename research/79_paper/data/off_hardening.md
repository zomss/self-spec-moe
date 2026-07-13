# OFF-region hardening — exhaustive combo search, optimistic pricing

Pricing: measured combo beta/R first; else product-law beta
(UPPER bound; measured exceptions are destructive) x term-edit R.
delivery = 1 (Phase-81 floor-free chain). kvq excluded (not
draft-only implementable). `!` = combo in the measured-destructive
family (mla, ctx>=32k, skip x context-lever) — its price is
optimistic by ~0.10-0.12 beta.


## mla (53 configs x gamma<=8 per cell)

### b4/2k — max 1.08x (q_fp8) -> **MARGINAL — measure**

| config | speedup* | gamma | beta | R | source |
|---|---|---|---|---|---|
| q_fp8 | 1.08x | 6 | 0.999 | 0.895 | prodB/measR |
| win512 | 1.06x | 2 | 0.967 | 0.871 | prodB/measR |
| skip125 | 1.02x | 1 | 0.966 | 0.930 | prodB/modelR |
| lr50+skip125 | 1.00x | 1 | 0.939 | 0.947 | measB/modelR |
| lr50+skip125+win512 | 0.99x | 1 | 0.930 | 0.945 | prodB/modelR |
| lr50 | 0.99x | 1 | 0.995 | 1.020 | prodB/modelR |
| lr25+skip125 | 0.99x | 1 | 0.922 | 0.947 | prodB/modelR |
| skip125+win512 | 0.98x | 1 | 0.899 | 0.928 | measB/modelR |

### b4/16k — max 1.02x (skip125) -> **OFF HOLDS**

| config | speedup* | gamma | beta | R | source |
|---|---|---|---|---|---|
| skip125 | 1.02x | 1 | 0.966 | 0.921 | prodB/modelR |
| lr50+skip125 | 1.00x | 1 | 0.942 | 0.935 | prodB/modelR |
| skip125+win512 | 0.99x | 1 | 0.891 | 0.901 | prodB/modelR |
| lr25+skip125 | 0.98x | 1 | 0.904 | 0.935 | prodB/modelR |
| lr50 | 0.98x | 1 | 0.975 | 1.016 | prodB/modelR |
| lr50+skip125+win512 | 0.98x | 1 | 0.868 | 0.915 | prodB/modelR |
| q_fp8+skip125 | 0.96x | 1 | 0.958 | 1.037 | measB/modelR |
| lr25 | 0.96x | 1 | 0.936 | 1.016 | prodB/modelR |

### b4/32k — max 1.07x (q_fp8) -> **MARGINAL — measure**

| config | speedup* | gamma | beta | R | source |
|---|---|---|---|---|---|
| q_fp8 | 1.07x | 5 | 0.995 | 0.909 | prodB/measR |
| skip125 | 0.99x | 1 | 0.902 | 0.913 | prodB/modelR |
| lr50 | 0.97x | 1 | 0.954 | 1.014 | prodB/modelR |
| q_fp8+skip125 | 0.95x | 1 | 0.902 | 1.011 | measB/modelR |
| lr50+q_fp8 | 0.93x | 1 | 0.953 | 1.092 | measB/modelR |
| lr25 | 0.93x | 1 | 0.878 | 1.014 | prodB/modelR |
| skip25 | 0.93x | 1 | 0.701 | 0.827 | prodB/modelR |
| lr25+skip125 | 0.93x | 1 | 0.792 | 0.925 | prodB/modelR |

### b8/2k — max 1.02x (skip125) -> **OFF HOLDS**

| config | speedup* | gamma | beta | R | source |
|---|---|---|---|---|---|
| skip125 | 1.02x | 1 | 0.966 | 0.924 | prodB/modelR |
| q_fp8 | 1.02x | 3 | 0.999 | 0.967 | prodB/measR |
| lr50+skip125 | 1.00x | 1 | 0.939 | 0.944 | measB/modelR |
| lr50+skip125+win512 | 0.99x | 1 | 0.930 | 0.940 | prodB/modelR |
| skip125+win512 | 0.99x | 1 | 0.899 | 0.920 | measB/modelR |
| lr25+skip125 | 0.99x | 1 | 0.922 | 0.944 | prodB/modelR |
| lr50 | 0.99x | 1 | 0.995 | 1.023 | prodB/modelR |
| lr25+skip125+win512 | 0.97x | 1 | 0.891 | 0.940 | prodB/modelR |

### b8/16k — max 1.03x (skip125) -> **OFF HOLDS**

| config | speedup* | gamma | beta | R | source |
|---|---|---|---|---|---|
| skip125 | 1.03x | 1 | 0.966 | 0.911 | prodB/modelR |
| lr50+skip125 | 1.01x | 1 | 0.942 | 0.926 | prodB/modelR |
| skip125+win512 | 1.01x | 1 | 0.891 | 0.880 | prodB/modelR |
| q_fp8 | 1.00x | 1 | 1.000 | 0.990 | prodB/measR |
| lr25+skip125 | 0.99x | 1 | 0.904 | 0.926 | prodB/modelR |
| lr50+skip125+win512 | 0.99x | 1 | 0.868 | 0.895 | prodB/modelR |
| lr50 | 0.98x | 1 | 0.975 | 1.017 | prodB/modelR |
| lr50+q_fp8+skip125 | 0.98x | 1 | 0.942 | 0.985 | prodB/modelR |

### b8/32k — max 1.05x (win512) -> **MARGINAL — measure**

| config | speedup* | gamma | beta | R | source |
|---|---|---|---|---|---|
| win512 | 1.05x | 1 | 0.813 | 0.725 | prodB/measR |
| skip125 | 1.00x | 1 | 0.902 | 0.903 | prodB/modelR |
| q_fp8+skip125 | 0.97x | 1 | 0.902 | 0.954 | measB/modelR |
| lr50 | 0.97x | 1 | 0.954 | 1.013 | prodB/modelR |
| lr50+q_fp8 | 0.95x | 1 | 0.953 | 1.046 | measB/modelR |
| q_fp8 | 0.95x | 1 | 0.995 | 1.106 | prodB/measR |
| skip25 | 0.94x | 1 | 0.701 | 0.805 | prodB/modelR |
| lr25+skip125 | 0.94x | 1 | 0.792 | 0.914 | prodB/modelR |

### b32/2k — max 1.07x (q_fp8) -> **MARGINAL — measure**

| config | speedup* | gamma | beta | R | source |
|---|---|---|---|---|---|
| q_fp8 | 1.07x | 5 | 0.999 | 0.909 | prodB/measR |
| skip125 | 1.03x | 1 | 0.966 | 0.913 | prodB/modelR |
| lr50+skip125+win512 | 1.01x | 1 | 0.930 | 0.905 | prodB/modelR |
| q_fp8+skip125 | 1.01x | 1 | 0.956 | 0.934 | measB/modelR |
| lr50+skip125 | 1.01x | 1 | 0.939 | 0.917 | measB/modelR |
| q_fp8+skip125+win512 | 1.01x | 1 | 0.934 | 0.921 | prodB/modelR |
| lr50+q_fp8+skip125+win512 | 1.00x | 1 | 0.929 | 0.923 | prodB/modelR |
| lr25+skip125 | 1.00x | 1 | 0.922 | 0.917 | prodB/modelR |

### b32/16k — max 1.13x (q_fp8) -> **MARGINAL — measure**

| config | speedup* | gamma | beta | R | source |
|---|---|---|---|---|---|
| q_fp8 | 1.13x | 7 | 1.000 | 0.850 | prodB/measR |
| skip125 | 1.04x | 2 | 0.966 | 0.893 | prodB/modelR |
| skip125+win512 | 1.03x | 1 | 0.891 | 0.831 | prodB/modelR |
| q_fp8+skip125 | 1.03x | 1 | 0.958 | 0.903 | measB/modelR |
| q_fp8+skip125+win512 | 1.03x | 1 | 0.891 | 0.841 | prodB/modelR |
| win512 | 1.03x | 1 | 0.922 | 0.873 | prodB/measR |
| lr50+skip125 | 1.02x | 1 | 0.942 | 0.895 | prodB/modelR |
| lr50+q_fp8+skip125 | 1.02x | 1 | 0.942 | 0.904 | prodB/modelR |

### b32/32k — max 1.04x (q_fp8) -> **OFF HOLDS**

| config | speedup* | gamma | beta | R | source |
|---|---|---|---|---|---|
| q_fp8 | 1.04x | 4 | 0.995 | 0.941 | prodB/measR |
| skip125 | 1.01x | 1 | 0.902 | 0.886 | prodB/modelR |
| lr50+q_fp8 | 0.98x | 1 | 0.953 | 1.000 | measB/modelR |
| lr50 | 0.98x | 1 | 0.954 | 1.002 | prodB/modelR |
| skip25 | 0.96x | 1 | 0.701 | 0.772 | prodB/modelR |
| q_fp8+skip125+win512! | 0.95x | 1 | 0.730 | 0.814 | prodB/modelR |
| q_fp8+win512 | 0.95x | 1 | 0.818 | 0.910 | measB/modelR |
| q_fp8+skip25 | 0.95x | 1 | 0.698 | 0.785 | prodB/modelR |


## moe (53 configs x gamma<=8 per cell)

### b4/2k — max 1.04x (lr50+q_fp8) -> **OFF HOLDS**

| config | speedup* | gamma | beta | R | source |
|---|---|---|---|---|---|
| lr50+q_fp8 | 1.04x | 1 | 0.836 | 0.768 | measB/modelR |
| lr50+q_fp8+win512 | 1.03x | 1 | 0.816 | 0.761 | measB/modelR |
| lr50+win512 | 1.03x | 1 | 0.821 | 0.775 | measB/modelR |
| q_fp8 | 1.02x | 2 | 0.990 | 0.952 | prodB/measR |
| lr50+q_fp8+win128 | 1.02x | 1 | 0.786 | 0.759 | prodB/modelR |
| lr50+win128 | 1.01x | 1 | 0.794 | 0.773 | prodB/modelR |
| lr50 | 1.01x | 1 | 0.837 | 0.826 | prodB/measR |
| win128 | 0.99x | 1 | 0.949 | 0.965 | prodB/measR |

### b8/2k — max 1.06x (q_fp8) -> **MARGINAL — measure**

| config | speedup* | gamma | beta | R | source |
|---|---|---|---|---|---|
| q_fp8 | 1.06x | 4 | 0.990 | 0.904 | prodB/measR |
| lr50+q_fp8 | 1.05x | 1 | 0.836 | 0.750 | measB/modelR |
| lr50+q_fp8+win512 | 1.05x | 1 | 0.816 | 0.737 | measB/modelR |
| lr50+q_fp8+win128 | 1.03x | 1 | 0.786 | 0.734 | prodB/modelR |
| q_fp8+win512 | 1.01x | 1 | 0.949 | 0.927 | measB/modelR |
| lr50+win512 | 1.01x | 1 | 0.821 | 0.807 | measB/modelR |
| q_fp8+win128 | 1.01x | 1 | 0.939 | 0.924 | prodB/modelR |
| lr50 | 1.01x | 1 | 0.837 | 0.826 | prodB/measR |

### b32/2k — max 1.07x (q_fp8+win512) -> **MARGINAL — measure**

| config | speedup* | gamma | beta | R | source |
|---|---|---|---|---|---|
| q_fp8+win512 | 1.07x | 2 | 0.949 | 0.825 | measB/modelR |
| lr50+q_fp8+win512 | 1.07x | 1 | 0.816 | 0.694 | measB/modelR |
| q_fp8+win128 | 1.07x | 2 | 0.939 | 0.816 | prodB/modelR |
| q_fp8 | 1.06x | 4 | 0.990 | 0.900 | prodB/measR |
| lr50+q_fp8 | 1.06x | 1 | 0.836 | 0.728 | measB/modelR |
| lr50+q_fp8+win128 | 1.06x | 1 | 0.786 | 0.685 | prodB/modelR |
| lr25+q_fp8+win128 | 1.00x | 1 | 0.679 | 0.685 | prodB/modelR |
| lr50 | 0.99x | 1 | 0.837 | 0.847 | prodB/measR |


## Measurement queue (challengers above 1.05x optimistic)

- mla b32/16k: q_fp8 priced 1.13x (gamma7, prodB/measR)
- mla b4/2k: q_fp8 priced 1.08x (gamma6, prodB/measR)
- moe b32/2k: q_fp8+win512 priced 1.07x (gamma2, measB/modelR)
- mla b4/32k: q_fp8 priced 1.07x (gamma5, prodB/measR)
- mla b32/2k: q_fp8 priced 1.07x (gamma5, prodB/measR)
- moe b8/2k: q_fp8 priced 1.06x (gamma4, prodB/measR)
- mla b8/32k: win512 priced 1.05x (gamma1, prodB/measR)

## Measured: the top challenger (mla b32/16k q_fp8) — COLLAPSES e2e; OFF stands

First MLA self-spec e2e in the record (`scripts/run_mla_challenger.sh`,
DeepSeek-V2-Lite TP1, fp8_per_block self-draft, shared KV, b32/16k):

| arm | tok/s | accept |
|---|---|---|
| nospec | 2452 ±675 | — |
| q_fp8 K=7 | 367 ±5 | **1.821** |
| q_fp8 K=5 | 413 ±4 | **1.802** |

The 1.13x optimistic price does not survive contact: measured 0.15x. TWO
independent failures, both informative:
1. **Accept 1.8 vs offline beta 0.995 (predicted accept ~6.3)** — the MLA
   e2e path was never anchor-gated (77's gate covered dense+moe only);
   either the MLA self-spec draft path has a defect (first exercise ever)
   or runtime fp8_per_block on V2-Lite diverges from the offline
   fake-quant beta. Backlog: MLA anchor-gate before any MLA map claim
   beyond OFF.
2. **Chain cost: decode 11.2s vs 1.77s nospec** — the MLA draft chain on
   this stack is far above its standalone-serve R (0.85); none of the
   Phase-81 chain fixes apply (no window -> no scratchpad graphs).

**Verdict: OFF on MLA is now measured, not just priced** — the best
challenger the exhaustive search could produce loses by 6.7x on the real
system. The optimistic pricing was an upper bound in the right direction.

## Anomaly RESOLVED: F11 strikes on FLASH_ATTN_MLA; fix applied; OFF triply measured

The accept collapse decomposed into three separately-measured causes:

1. **Harness defect (the big one, +4.1 accept): FLASH_ATTN_MLA's captured
   chain graph is not replay-safe** — it is FA3-family, so the
   constant-geometry law (81-F11) applies; the fg2-era "MLA keeps the
   captured graph" rule was a FLASHMLA-only fact. bf16 self-draft accept:
   captured 2.00 → eager 5.89 → **5.916 by default after the fix**
   (llm_base_proposer: FlashAttnMLAMetadataBuilder classified
   non-replay-safe). FLASHMLA itself untestable here (block-size
   incompatibility) — noted, not assumed.
2. **Lever mismatch (−2.1 accept)**: the harness's fp8_per_block draft is
   W8A8 (activations too); the map's q_fp8 β=0.995 is weight-only. W8A8 on
   V2-Lite: accept 3.86 at K5 (β_eff ≈ 0.77). A weight-only draft option
   is future plumbing.
3. **Chain cost is the binding loss anyway**: on the fixed chain, even the
   PERFECT-accept bf16 self-draft (5.916) reaches only 1364 tok/s = 0.56×
   nospec (2452 ±675) — the eager MLA chain step costs far above the
   standalone-serve R=0.85 the pricing used, and no Phase-81 remedy
   applies without a window (no scratchpad geometry to clamp).

**Final verdict: OFF on MLA at NVLink is (a) search-backed (max 1.13×
optimistic over 53 configs), (b) mechanism-explained, and (c) measured —
the best challenger delivers 0.54×, and even β≈1 delivers 0.56×.** The gap
to close before ANY MLA config can win is chain execution (captured MLA
chain via FLASHMLA or an MLA scratchpad), and even at perfect delivery the
priced ceiling is 1.13× — below every dense/MoE map winner.
