# Are window and skip badly implemented, or badly suited? Both, separably

Record of 2026-08-16. Analysis over the measured prediction map; no new
measurement. Script: `scripts/analyze_w98_lever_mechanics.py`; record:
`data/g98_e/lever_mechanics.json`.

Window-only aggregates to 1.01x and skip-only to 1.02x against
quantization's 1.20x, which invites the reading that our sparse-attention
and layer-skip implementations are simply bad — the literature reports both
working as self-speculative levers. The map carries the two quantities that
separate the hypotheses per cell and per regime: measured draft-chain cost
`d_hat` and measured acceptance `tau_hat`.

## The baseline nobody states: an unmodified draft LOSES

| | R1 | R4 | R5 | R5cot | R6 | R8 |
| --- | --- | --- | --- | --- | --- | --- |
| unmodified draft, speedup vs OFF | 0.891 | 0.923 | 0.938 | 0.938 | 0.939 | 0.911 |

Drafting with an unmodified copy of the target costs a full extra forward,
so self-speculation starts **8-11% underwater**. Every lever's first job is
to claw back to 1.000; only then does it add value. "Window is worth 1.01x"
is measured from OFF, not from the draft, and hides that the lever has
already paid for the draft chain.

## Skip: the implementation is fine, the operating point was not

**Cost is near-ideal.** Removing `n` of 36 layers should cut draft cost to
`1 - n/36`:

| | measured | ideal | overhead |
| --- | --- | --- | --- |
| skip 4 | 0.894-0.902 | 0.889 | +0.6 to +1.5% |
| skip 8 | 0.789-0.803 | 0.778 | +1.4 to +3.2% |

So the skip path delivers essentially the full layer-proportional saving.
There is no implementation defect on the cost side.

**The ceiling is arithmetic.** 4 of 36 layers buys an 11% cost cut and 8 buys
21%. That is simply not enough to overcome an 8-11% deficit and then win
big, whatever the acceptance. Going deeper is what D2(b) tested: at 16 of 36
every arm collapses to tau 1.0-2.1 against 6.4-7.4 at 8. **Training-free
skipping hits the acceptance wall before it reaches a useful cost
reduction**, and that is a property of the method under our constraint, not
of the code.

**But the layer SET was leaving a token on the table.** D2(b)'s reference arm
measured the frozen nested skip-8 set the cost campaigns actually booted at
tau **5.327**, against **6.423** for a knapsack-chosen set of the *same
count* — **+20.6% acceptance at identical cost**. Re-scoring skip-8 under
that uplift:

| skip 8 | R1 | R4 | R5 | R5cot | R6 | R8 | regimes won |
| --- | --- | --- | --- | --- | --- | --- | --- |
| frozen set (as evaluated) | 0.900 | 0.703 | 0.784 | 0.938 | **1.036** | 0.867 | **1 / 6** |
| knapsack set (estimate) | **1.086** | 0.848 | 0.945 | **1.131** | **1.249** | **1.045** | **4 / 6** |

*Estimate, not measurement.* The +20.6% factor was measured on one regime's
retention probe and is applied uniformly here; the direction is measured,
the per-regime magnitude is not.

> **MEASURED, AND THE ESTIMATE WAS WRONG (G98-F, `results_g98_f_skipsets.md`).**
> This section originally read that the estimate above was "a qualitative
> change, not a marginal one", that skip goes from worthless alone to beating
> no-speculation in four of six regimes, and that "window and skip are
> worthless alone" was therefore an artifact of the operating point and
> **withdrawn for skip**. That withdrawal is **reversed**.
>
> G98-F measured the knapsack sets out of sample. Two things came out:
>
> * **The acceptance uplift is real and it TRANSFERS.** Sets chosen from R8
>   seed-4 retention beat the frozen sets in **5 of 6** regimes on seeds 2-3
>   -- R4 +16.9%, R8 +15.2%, R5 +11.0%, R1 +8.1%, R5cot +7.4%, R6 -6.2% --
>   at unchanged cost. The knapsack sets should replace the frozen ones as the
>   skip baseline.
> * **It does not make skip a standalone lever.** Knapsack skip-8 beats OFF in
>   **2 of 6** regimes, not the estimated 4; the frozen set manages 3. Better
>   acceptance is not enough, because skip-8 still costs ~79% of a base draft
>   that itself loses to OFF at 0.891x.
>
> The error was extrapolating a single-regime factor uniformly and recomputing
> speedup from the FITTED cost map rather than measuring. **"Skip is worthless
> alone" largely stands**; what the knapsack sets buy is bigger wins where
> skip already won (R5cot 1.050 -> 1.135, R8 1.024 -> 1.174), not new ones.

## Window: physics first, then a small real overhead

**Where there is nothing to trim, windowing costs a little.**

| window cost fraction | R1 (short) | R6 (short) | R8 (short) | R4 (8k) | R5 (14k) | R5cot (14k) |
| --- | --- | --- | --- | --- | --- | --- |
| w1024 | **1.020** | **1.016** | **1.015** | 0.742 | 0.632 | 0.622 |
| w128 | **1.013** | 0.962 | 0.978 | 0.707 | 0.605 | 0.596 |

Ratios **above 1.0** mean the windowed draft is *slower* than the unwindowed
one. At short context there is no KV to skip, so the masking and bookkeeping
are pure overhead — **1.5-2%**, small but real, and it is the one genuine
implementation cost this analysis finds on the window path.

**Where there is something to trim, the lever is strong.** At 14k the same
lever cuts draft cost by **37-40%** and reaches **1.28-1.34x** over OFF. The
aggregate 1.01x averages a lever that cannot work at three regimes with one
that is the best available at two.

**And windowing is content-sensitive, sharply.** Acceptance retained under
`w1024`:

| R4 (8k, summarization) | R5 (14k, RAG QA) |
| --- | --- |
| **0.535** | **0.957** |

The *shorter*-context regime is destroyed by windowing while the longer one
is barely touched. Length alone predicts the opposite, so this is the task:
summarizing a document needs the whole document in view, and a windowed
draft cannot follow the target's next token. Retrieval QA attends locally
around the answer span and tolerates the window. This is a sharper content
result than the R5/R5cot pair (Spearman +0.902) that section 3.4 of the
summary rests on, and it should replace it as the content exhibit.

## Verdict on the two hypotheses

| | window | skip |
| --- | --- | --- |
| implementation defect | **yes, minor** — 1.5-2% overhead where it cannot help | **no** — cost is within 1-3% of the layer-proportional ideal |
| operating point wrong | no — all four widths measured | **YES** — frozen sets cost 20.6% acceptance at k=8 |
| inherent to the method | **yes** — gated by context length AND task | **yes** — training-free depth caps the cost cut before acceptance collapses |

Neither lever is broken. Skip was measured below its potential and is worth
re-running; window is a context- and content-gated lever whose aggregate
figure is an averaging artifact.

## On the literature comparison

Published results showing layer skipping or sparse attention carrying
self-speculation on their own generally sit outside at least one of our
constraints — they adapt or train the draft path (layer-dropout and
early-exit objectives make aggressive depth survivable), or they operate on
model families with more depth redundancy than Qwen3-8B shows here. Our
constraint is training-free and distribution-preserving, and D2(b) puts a
number on what that costs: acceptance collapses between 8 and 16 skipped
layers of 36. We have not audited those papers' protocols against ours, so
this is the reading our data supports rather than a refutation of theirs.

The one head-to-head we do run points the other way: against KnapSpec's
published 1.43x, our composed configuration reaches **1.57x** (`paper/c1.md`).
Our skip-*alone* is weaker than theirs; our composition is stronger.

## Recommended action

**Re-run the skip cells with knapsack-chosen layer sets.** ~~Two claims in the
record change if the estimate holds.~~ **DONE (G98-F).** Of the two claims
predicted to change, one did and one did not:

* ~~"window and skip are worthless alone" becomes false for skip~~ — **NO.**
  Knapsack skip-8 beats OFF in 2 of 6 regimes against the frozen set's 3. The
  claim stands.
* ~~the composition premium IS overstated~~ — **NO, and the direction was
  backwards too.** The +12.5% is `composed 1.35x / best single 1.20x`; the
  best single is **quantization**, which no layer set touches. The winning
  composition is `w4a16-quantized/w1024/skip4`, using count **4** — and at
  count 4 the knapsack set is not better (R1 -0.75%, R4 +0.63%, R5 -3.18%,
  R5cot 0.00%, R6 -4.26%, R8 -1.41%). The knapsack sets improve skip at k=8
  only, and neither term of the decomposition depends on k=8. **The premium
  stands at +12.5% and needs no recomputation.**
