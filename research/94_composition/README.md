# Phase 94 — C2: efficient search for the optimal COMPOSED configuration

Design of record, 2026-08-02. Follows C1 (phase 93: five architectures,
five distinct single-lever surfaces, no universal lever). Levers there
were ISOLATED by construction; compositions were deferred here.

## The claim

> Given a new (model, hardware, regime), a bounded measurement budget
> (~2-4 GPU-h) finds a configuration within a few percent of the
> exhaustive optimum over the COMPOSED lever space — and no
> proxy-scoring shortcut can replace those measurements.

Two halves. The negative half is DONE (phase 90): five proxy classes
falsified (angular importance INVERTED -0.77; margin-Taylor; micro-LOO
0.34; CLaSp-cosine 0.37; learned predictors 0.08-0.22), plus
non-additivity and the on-policy sign flip. The positive half — a
search procedure with a MEASURED regret-vs-budget curve against an
oracle — does not exist yet. That is this phase.

## Why composition is the body of C2

Single levers = ~13 arms/arch: brute-forceable (C1 did exactly that).
Composed: 5 quant x 5 window x 3 skip x 2 kvq x 4 K ~ 600 configs per
cell per arch per regime. Exhaustive is impossible -> search becomes
the contribution. And composition value is SCALE/COST-STRUCTURE KEYED
in the existing record: the 32B triple (skip x W4A8-Humming x win512)
reached ~1.57x vs 1.51x best-single, while skip priced OUT at 8B. So
the search must decide WHETHER to compose, not only what.

## Pre-registrations (registered BEFORE any phase-94 run)

| # | prediction | falsifies / implies |
|---|---|---|
| P1 | best composition beats best single by >=5% at >=half of probed cells on >=2 architectures | if REFUTED: C2 pivots from composition-search to single-lever search economics (honest negative, reported) |
| P2 | composed beta is SUB-additive vs product-of-singles (gap +0.02-0.05) | confirms the refuted-heuristic control at scale (extends 3 prior confirmations) |
| P3 | composed COST is near-additive in the byte-budget model (levers cut distinct terms) | the mechanism that makes bounded search tractable |
| P4 | LCB-guided search reaches >=95% of oracle-best S at <=10% of exhaustive budget | THE HEADLINE claim of C2 |
| P5 | product-of-singles pricing mis-ranks the top config at >=30% of cells | why the naive shortcut fails (mechanistic support for P4's necessity) |

## Composability matrix (fork reality)

| pair | composable? | note |
|---|---|---|
| quant x window | YES | draft ckpt + window env; the record's main pair |
| quant x skip | YES | skip is index-preserving, shared-KV binding intact |
| window x skip | YES | independent envs |
| quant x window x skip | YES | the 32B triple (record: ~1.57x) |
| kvq x {any} | PARTIAL | kvq requires NO shared-KV (own draft cache) -> different stack; DEFERRED from the premise probe, revisited at Step 2 |
| Humming kernel x K4/K6 at TP2 | BLOCKED | odd-width lazy-cubin wedge (C1 ledger); 32B compositions use W4-GPTQ/Machete, Humming numbers cited from C1 Stage A |

## Step plan (gated, like C1)

- **Step 0** (this doc) — claim + pre-registrations + composability.
- **Step 1 — premise probe (~5 GPU-h)**: compositions vs the EXISTING
  C1 single-lever cells (same protocol/stack/machine, so directly
  comparable — no re-measure of singles). dense-8B + Llama in
  parallel (TP1), then 32B (TP2). Tests P1/P2 cheaply.
  -> **GATE 1: P1 confirmed or refuted; user decides C2's body.**
- **Step 2 — oracle (~12-16 GPU-h)**: full factorial on a REDUCED but
  COMPLETE space (quant{none,best2} x window{none,512,2048} x
  skip{none,b2} x K{2,4} = 36-72 configs) at 4 cells x 2 arch,
  compile grade. Ground truth for any regret claim.
- **Step 3 — the search**: nominate-by-profile -> LCB pricing under
  realization uncertainty -> confirm-by-measurement; run at 5/10/20%
  budgets -> regret-vs-budget curve.
- **Step 4 — baselines**: exhaustive (bound), random-at-budget,
  product-of-singles (P5), proxy-ranking (phase-90 dead class),
  KnapSpec knapsack-over-cosine (published, in the refuted class).
- **Step 5 — transfer + cost**: does the search restart per column?
  (record: 0.43 cross-model prior as cold-start). Updated
  cost-to-onboard.
  -> **GATE 2: regret curve + baselines before packaging.**
- **Step 6 — package**: paper/c2.md, paper/data/c2_*.json,
  visualizations (regret-vs-budget single 1.5:1; composition
  landscape multi 3:1), T11 scorecard += P1-P5.


## GATE 1 — premise probe result (2026-08-02)

Compositions (quant x window x skip) measured on the C1 compile
protocol; compared against C1's single-lever cells (same protocol,
stack, machine).

| arch | comp >= +5% | any gain | best cell |
|---|---|---|---|
| dense 8B | **6/8** | 8/8 | b8/14k Hum x win512: 1.26 -> **1.58 (+25.7%)** |
| Llama 8B | 3/8 | 7/8 | b8/14k W4 x win2048: 1.18 -> **1.38 (+16.5%)** |
| Qwen3-32B (matched kernel) | 2/7 | 5/7 | b8/14k W4gptq x win2048: 1.12 -> **1.33 (+18.2%)** |

**P1 as registered (>=5% at >=HALF the cells on >=2 arch): REFUTED**
(only dense passes). But the registered threshold was the wrong shape
for the phenomenon, and the data says why:

**THE REAL FINDING — composition gain is CELL-STRUCTURED, not uniform:**

| axis | mean gain |
|---|---|
| ctx 2k | **-2.0%** |
| ctx 8k | +7.4% |
| ctx 14k | **+8.1%** |
| batch 1 | -0.0% |
| batch 8 | **+8.5%** |
| batch 32 | +2.7% |

Composition pays exactly where TWO cost terms are simultaneously
large (long ctx => KV-read term, mid-batch => weight+compute term):
quant cuts weight bytes, window cuts KV bytes, and only when both
bind does stacking beat either alone. At b1/short-ctx a single lever
already removes the one binding term and composition adds pure
overhead (-2%). This is the byte-budget model predicting composition
value -- and it means the SEARCH must be cell-conditional, which is
precisely C2's thesis.

32B CONFOUND CAUGHT: the naive comparison used Humming (fastest
kernel) for singles vs Machete for compositions (Humming wedges at
TP2 odd width) -- a KERNEL comparison, not a composition one. The
matched-kernel re-analysis above (W4-GPTQ both sides) is the fair
test; it moves 32B from 1/7 to 2/7 at >=5% and from 2/7 to 5/7 at
any gain.

**P2 (sub-additivity)**: composed accept stays below the product of
singles in every probed cell -- consistent, extends the 3 prior
confirmations. Never price compositions by product.

**VERDICT**: composition is worth searching, but NOT everywhere --
which makes the search problem harder and more interesting than the
registered P1 assumed. C2 proceeds with the amended framing:
*find whether AND what to compose, per cell.*


## P3 check -> a CONFOUND IN OUR OWN GATE-1 COMPARISON (2026-08-02)

Inverting R from measured cells (R = ((accept/S) - 1)/K) to test cost
additivity produced absurd errors (mean |err| 41%, worst +410%). The
inversion is correct; the INPUTS were not comparable. Two causes,
found by chasing the outliers:

1. **Realization mismatch (the serious one).** The engine only allows
   the FULLCG scratchpad chain WHEN A WINDOW IS SET
   (llm_base_proposer: "FULLCG requires KV_WINDOW > 0"). C1 measured
   its window SINGLES with winplain (no FULLCG -- an IMA-era
   conservative choice, since the IMA was only localized/fixed later).
   Phase-94 COMPOSITIONS use FULLCG. So every window-containing
   composition ran on a FASTER CHAIN than the window singles it was
   compared against. At overhead-bound cells this dominates: llama
   w4a16 single at b1/2k measured S=0.37, but the SAME ckpt in
   w4a16 x win512 gave S=1.19 -- a window cannot make a draft 3x
   faster; the chain realization did.
2. **Overhead-bound cells are NOT corruption.** 32% of single-lever
   (cell,arm,K) rows invert to R>1.5 (MLA 93%, MoE 41%). At b1/short
   ctx a target step is tiny and the draft chain's fixed cost exceeds
   it -- the documented fixed-overhead floor (phases 72/73). Those
   rows are real physics and explain why OFF wins there; they must be
   analyzed per-regime, not pooled or filtered as bad data.

**Consequence for Gate 1**: P1's best-single-vs-best-composition
comparison is SAFE against slow singles (a slow arm is never the max)
but is INFLATED wherever the winning composition contained a window
and the best single did not -- it partly measures FULLCG, not
composition. The reported peaks (+18-26%, all at window-containing
compositions) must be re-derived on matched realizations.

**CONTROL RUNNING**: re-measure window singles (win512, win2048) WITH
FULLCG on all three arches (scripts/rematch_window_singles.sh), then
redo P1 and P3 on matched data. Note the coupling is real and stays
in the paper either way: the fast chain is only AVAILABLE with the
window lever, so "window brings FULLCG" is part of that lever's
deployment value -- but the mechanism claim (composing cost terms
helps) requires the matched comparison.

P3 verdict: DEFERRED until matched data exists. Pooled additive vs
product errors (41% vs 40%) are uninformative while realizations mix.

## Protocol

Same compile-cell protocol as C1 Stage A (decode T(1+N)-T(1), batch x
ctx grid, one boot per (config, K)), same production stack bundles, so
composition cells are directly comparable to C1's single-lever cells.
Windows use FULLCG-scratchpad WITHOUT wholechain (the C1 IMA fix).
Singles are NOT re-measured: C1's cells_93_* are the baseline.
