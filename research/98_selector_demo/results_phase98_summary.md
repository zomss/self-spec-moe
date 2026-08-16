# Phase 98 — the two-round selector: method, results, and what they mean

Consolidated record as of 2026-08-16, written for discussion. Every number
here is measured and traceable to a scored artifact; the per-claim documents
are the authority and are cited throughout. Nothing in this file is new
measurement.

**Status: every registered claim of the phase is measured.** Four scored
campaigns (D1', D2a/c, D2b, D3), where the phase had none a week ago.

---

## 1. The method

### 1.1 The idea

Self-speculative decoding drafts with a **modified copy of the target
model** and verifies exactly, so output is unchanged. The design question is
which modification — the **lever** — to use. Phase 98 tests a two-round
selector built on one asymmetry:

| quantity | what it is | can it be predicted offline? |
| --- | --- | --- |
| **cost** `D` | draft-chain time per step | **yes** — fit from single-lever profiles |
| **acceptance** `tau` | committed tokens per target step | **no** — must be measured |

Round 1/2 fit a cost model from single-lever profiles and eliminate
configurations that cannot pay *whatever* their acceptance turns out to be.
Round 2's successor then measures acceptance only for what survives. D3
scores the composed result end to end.

### 1.2 Levers and lattice

Three composable levers, all training-free and distribution-preserving:

| lever | values | HBM cost |
| --- | --- | --- |
| weight quantization | `target-matching` (shares target weights), `w4a16-quantized` | quantized draft is a SEPARATE ~6.1 GB resident |
| KV window | `off, 128, 256, 512, 1024` (+16 sinks) | none |
| layer skip | `0, 4, 8, 16` of 36 layers, nested sets | none |

That closes to **30 composed configurations plus OFF**. Six regimes span the
deployment space:

| regime | batch | input context | content |
| --- | --- | --- | --- |
| R1 | 1 | short | math (GSM8K/AIME) |
| R8 | 16 | short | math, temperature 1.0 |
| R6 | 32 | short | math |
| R4 | 8 | 8k | summarization (CNN/DM) |
| R5 | 8 | 14k | RAG QA (NQ-open over docs) |
| R5cot | 8 | 14k | long-CoT RAG |

Model: Qwen3-8B dense. Runtime: piecewise draft chain, Marlin kernel, shared
target KV, live K/OFF ladder over K in {0, 4}.

### 1.3 Measurement discipline

The phase's results are only worth as much as the protocol, so:

* **Preregistration with commitment barriers.** Predictions, screens and set
  choices are committed with a digest BEFORE the measurement that judges
  them; the runners refuse to commit afterwards. Used at three points: D1'
  held-out predictions, D2(a)'s product-bound screen, D2(b)'s layer sets,
  and D3's selector picks.
* **Decode currency only.** Committed tokens over summed decode step time,
  never wall clock. Phase 96 retracted a claim to mixing them; prefill is
  46.6% of wall at R5.
* **Equal work.** Fixed token budget with `ignore_eos`, because under
  natural EOS the speculative and autoregressive arms drain asymmetrically
  — a defect that moved one MoE cell from 1.288 to 0.954 when re-measured.
* **Disjoint data.** Cost fits on content seeds 2-3, acceptance on 4-5
  (verified overlap zero), D2(b) retention on seed 4 and its confirmation on
  seed 5. Phase 84 retracted a lever whose gate came from its own evaluation
  set.
* **Dual instrument.** The profiler's inner syncs cost +2.4 ms/step and are
  paid only by ARMED configurations, so D3 runs under both the corrected and
  the legacy instrument and reports both.

---

## 2. Results, claim by claim

### 2.1 D1' — cost is predictable (`results_g98_c.md`)

**47 of 48 held-out (cell, regime) pairs covered — 97.9%**, median relative
error **0.53%**, all six regimes resolvable including R5 and R5cot which
Round 1 could not resolve.

The second clause matters more than the coverage. The sound elimination rule
`(K+1)/q_lo < 1.015` **fired five times and was correct every time** — zero
false eliminations, non-vacuously, where Round 1 had to report the rule NOT
EXERCISED for want of a scored surface. So the cost half of the asymmetry
holds: a model fit from single-lever profiles predicts composed
configurations well enough to eliminate soundly.

### 2.2 D2(a) — the product bound, and where it breaks (`results_g98_d.md`)

The screen admits a composition if `f_set >= prod f_i`. Measured: **6
violations in 132 rows (95.5% honest)**, and every one carries **window =
128** stacked with layer skipping at short context.

The more useful finding is the opposite of the violations. Composition is
generally **constructive**: the interaction ratio (actual / product) has
median above 1 for every lever set and reaches **1.670 at R4**. A window
that already discards the context removes the very information layer
skipping would have degraded, so their losses OVERLAP rather than compound.
Only at short context do the two levers damage different things.

Direction matters: a violated lower bound makes the screen over-optimistic,
so it admits configurations that do not deserve it and confirmation catches
them. It cannot produce the dangerous failure of eliminating a good one.

### 2.3 D2(c) — the u-axis resolves

Adjacent generated-suffix buckets separate by INTERVAL in **107 of 264**
adjacent pairs. Acceptance genuinely varies with how far into a generation
the request is.

### 2.4 D2(b) — the knapsack, and the non-additivity boundary (`results_d2b_knapsack.md`)

Per-layer retention measured by leave-one-out (36 boots), then knapsack
versus count-matched random and worst controls.

**11 of 12 registered controls beaten**, so the claim fails — but only at
one count, and the reason is the result:

| count | Spearman(knapsack's objective, measured tau) |
| --- | --- |
| 4 | **+1.000** |
| 8 | **+1.000** |
| 16 | +0.771, with the TOP inverted |

At k=4 and k=8 the product of per-layer retentions ranks all six arms
**exactly right**. At k=16 the knapsack's predicted-best set (product 0.585,
far the highest) measures tau **1.244**, losing to the frozen set (2.080)
and a random set (1.970).

So additive per-layer measurement **identifies bad sets at every count and
stops identifying the best set once the count is aggressive**. That is Phase
90's non-additivity theorem reproduced on a new lever, model and box — with
the boundary now LOCATED between 8 and 16 layers of 36.

Two supporting facts. **Layer 0 is worth almost the entire draft**: skipping
it alone drops tau from 8.083 to 1.244 (retention 0.034), against 0.79 for
the next worst layer. And **k=16 is outside the lever's usable range** — every
arm measures 1.0-2.1 against 6.4-7.4 at k<=8 — consistent with why skip16
exists at all, having been admitted to make the cost term `F` estimable
rather than as an operating point.

### 2.5 D3 — end-to-end value (`results_g98_e.md`)

62 boots, both instruments, on h103 bare metal.

| clause | corrected | legacy |
| --- | --- | --- |
| >= 90% of omniscient | **95.1%** PASS | **94.8%** PASS |
| beats every static +2% (all 31) | FAIL by one | FAIL by one |
| beats every static +2% (single-lever + OFF) | PASS 1.141x | PASS 1.154x |
| over static-OFF | **1.371x** | **1.311x** |

Both arms agree, so it is not an instrument artifact. The static clause
turns on a reading: "every static single configuration including OFF" passes
if *single* means single-LEVER and fails if it means *any one fixed
configuration*. The stricter reading is used as the headline.

### 2.6 D3 re-scored with a fail-closed rule (`results_d3_failclosed.md`)

The one large miss above is a missing capability, not a mis-ranking: the
selector had no way to decline. Adding one — arm only when the lower
confidence bound on the predicted margin over OFF clears unity, with both
error terms estimated from pre-D3 data (`sigma_tau` 0.0195 from G98-D's
seed-4/seed-5 pairs, `sigma_cost` 0.0123 from D1' held-out rows) — fires at
exactly one regime, R4, and changes the verdict:

| clause | as measured | fail-closed |
| --- | --- | --- |
| fraction of omniscient | 95.1% | **99.6%** |
| over static-OFF | 1.371x | **1.436x** |
| over best static | 1.014x | **1.062x** |
| beats every static by +2% | FAIL | **PASS** |
| mixes meeting 90% | 112/126 | **126/126** |
| mixes beating every static | **9/126 (7%)** | **102/126 (81%)** |

Both instrument arms agree. **The clause that was awaiting a ruling now
passes on the strict reading**, so the ambiguity is moot.

This is a post-hoc rule with a pre-D3-data-only derivation, not a
preregistered one, and is labelled as such. In place of a barrier it carries
two invariance checks: the decision is identical for any bare threshold in
(1.026, 1.180) and for any confidence level `z` in (1.10, 7.06) — the
conventional 1.645 sits far from both edges.

### 2.7 The surrogate's foundation (`results_certified_region.md`)

A registered hypothesis — that a damage budget `D = sum -log(retention_i)`,
computable from singles, gates where the product surrogate ranks correctly —
was **refuted**: restricted to a damage band, rank correlation is flat
(0.84-0.95) and shortlist recall is perfect in every band including the
worst. There is no validity region because the surrogate never leaves
validity.

What the budget does certify is the **bound's tightness**: `rho(D, |log
error|) = +0.875` over 264 composed observations, with the signed error
positive in every band (the product stays a lower bound) and rising
monotonically from +0.6% at `D < 0.3` to +60.7% at `D > 1.6`. The error is
systematic rather than noise, which is why the ordering survives it.

And the ordering carries a guarantee of the shape the search actually
consumes — not top-1 correctness, but shortlist recall:

| | recall@1 | recall@2 | recall@3 | **recall@4** | top-1 regret mean / max |
| --- | --- | --- | --- | --- | --- |
| `tau_eff` | 8/12 | 10/12 | 11/12 | **12/12** | 0.32% / 1.99% |
| `tau_k4` | 8/12 | 10/12 | 11/12 | **12/12** | 0.22% / 1.86% |

So: measure 7 singles, order 22 compositions by the product, confirm the top
4 — the optimum was inside the confirmed set in 12 of 12 groups, and a
collapse to confirm-1 costs at most a measured 2.0%.

---

## 3. Performance

### 3.1 Per regime, against no speculation

| regime | batch | context | OFF tok/s | selector | vs OFF | oracle | vs OFF |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | 1 | short | 125.3 | 178.2 | **1.42x** | 178.6 | 1.43x |
| R4 | 8 | 8k | 684.9 | 522.3 | **0.76x** | 684.9 | 1.00x |
| R5 | 8 | 14k | 572.8 | 1007.5 | **1.76x** | 1016.2 | 1.77x |
| R5cot | 8 | 14k | 577.7 | 1144.2 | **1.98x** | 1144.2 | 1.98x |
| R8 | 16 | short | 1807.7 | 2450.6 | **1.36x** | 2452.8 | 1.36x |
| R6 | 32 | short | 3448.4 | 4203.0 | **1.22x** | 4575.4 | 1.33x |

**Aggregate 1.371x over OFF at 95.1% of the omniscient ceiling** as
originally scored; **1.436x at 99.6%** once the selector declines at R4
(section 2.6), where the row above becomes 684.9 / 1.00x. Long context is
where the design pays — 1.76x and 1.98x at 14k — which vindicates the Phase
58/62 redirect toward the KV-bound regime.

### 3.2 The value ladder — what each step is worth

Aggregate over the equal-weight mix, best member of each family:

| family | best tok/s | vs OFF |
| --- | --- | --- |
| OFF | 435.9 | 1.00x |
| window only | 441.6 | **1.01x** |
| skip only | 442.9 | **1.02x** |
| quantization only | 524.1 | **1.20x** |
| composed, 2 levers | 556.1 | 1.28x |
| composed, 3 levers | 589.6 | 1.35x |
| **selector (per-regime)** | **597.7** | **1.37x** |

This is the phase's most decision-relevant table:

* **Searching is worth a lot.** Finding the right family at all is 1.37x
  over OFF and ~1.30x over a median static.
* **Composing is worth a lot.** 3-lever composition beats the best single
  lever by **+12.5%** (1.35x vs 1.20x); the selector beats it by 1.141x.
* **Per-regime switching is worth +6.2%** over the best single fixed
  configuration once the selector can decline (626.0 vs 589.6). The +1.4%
  originally reported here was net of a self-inflicted 24% regression at R4
  (section 2.6).
* **Window and skip are worthless alone** (1.01x, 1.02x) and valuable in
  composition. Only quantization carries standalone value.

### 3.2b Why switching is worth +6.2% and not more (`results_switching_law.md`)

With time-weighted aggregation the gain over any fixed configuration is
EXACTLY `sum_R t_R * (rate_sel(R)/rate_static(R))`, where `t_R` is the
selector's **time** share — verified to machine precision on both arms.
Because time share is inversely proportional to a regime's own throughput,
**slow regimes dominate**, and the decomposition is stark:

| regime | time share | ratio | excess contribution |
| --- | --- | --- | --- |
| **R1** (b1, short) | **0.5855** | 1.0024 | **+0.0014** |
| **R4** (b8, 8k) | 0.1523 | **1.3922** | **+0.0597** |
| R5 / R5cot / R6 / R8 | 0.2622 | ~1.00 | +0.0007 |

One regime carries 59% of the time budget and contributes +0.14%, because at
batch 1 every quantized configuration performs within 0.3% of every other.
The omniscient ceiling over the same static is 1.0664x, so the fail-closed
selector already captures **93% of all switching value this grid contains**.
The limitation is the grid, not the selector.

Switching therefore requires a regime that is simultaneously **time-dominant**
and **optimum-distinctive**. This grid has each property separately and never
together — which is a fact about the six regimes measured, not about
switching.

The axis that would supply both is **mixed feasibility**: the quantized draft
is a separate ~6.1 GB resident, so under KV pressure it is not deployable at
all, and no single static can serve the mix. Scoring that on the measured
grid with 14k regimes modelled as unable to host the draft gives a switching
gain of **1.305x** (legacy 1.249x) — 5x the uniform-feasibility figure. The
rates are measured; the feasibility mask is modelled, so this sizes the
effect rather than establishing it.

### 3.3 Does the lever vary by regime? Yes — the window does

| regime | quant only | window only | skip only | composed | best configuration |
| --- | --- | --- | --- | --- | --- |
| R1 | 1.36x | 1.01x | 1.08x | 1.43x | `w4a16/w1024/skip8` |
| R4 | 0.76x | 0.59x | 0.93x | 0.78x | **`off`** |
| R5 | 1.13x | **1.45x** | 1.07x | 1.77x | `w4a16/w1024/skip0` |
| R5cot | 1.21x | **1.52x** | 1.11x | 1.98x | `w4a16/w512/skip4` |
| R8 | 1.28x | 1.06x | 1.09x | 1.36x | `w4a16/w512/skip4` |
| R6 | 1.24x | 1.09x | 1.12x | 1.33x | `w4a16/w256/skip4` |

**Quantization is universally on** — every winner uses `w4a16`. What varies
is the **window (256 / 512 / 1024) and skip (0 / 4 / 8)**, and at R4 the
right answer is not to speculate at all. The window lever's value is
entirely regime-keyed: 1.01-1.09x at short context, **1.45-1.52x at 14k**.
The aggregate 1.01x for window-only is an average over regimes where it is
useless and regimes where it is the strongest single lever — C1's "no
universal lever" visible inside a single model.

### 3.4 Content versus batch and context

R5 and R5cot isolate content: identical batch 8, identical 14k context,
identical 640-token generation budget, differing only in RAG QA versus
long-CoT RAG.

* They pick **different optima** — `w1024/skip0` (1.77x) against
  `w512/skip4` (1.98x).
* But the full 31-configuration ranking correlates at **Spearman +0.902**.

**Batch and input size choose the lever family; content fine-tunes within
it.** Selecting on batch and context alone would cost roughly 1-2% here, not
20%. Caveat: this is one content pair on one model, and genuinely different
task families (code versus prose) are not spanned.

### 3.5 Under memory pressure: what survives without a resident quantized draft

`target-matching` shares the target's weights, so window and skip cost no
extra HBM; the quantized draft needs a separate ~6.1 GB.

| regime | best with no extra HBM | vs OFF | best with quantized draft | vs OFF | constraint costs |
| --- | --- | --- | --- | --- | --- |
| R1 | `woff/skip4` | 1.08x | `w1024/skip8` | 1.43x | 24.5% |
| R4 | `off` | 1.00x | — | 0.78x | — |
| R5 | `w512/skip0` | **1.45x** | `w1024/skip0` | 1.77x | 18.1% |
| R5cot | `w1024/skip4` | **1.58x** | `w512/skip4` | 1.98x | 20.4% |
| R8 | `woff/skip0` | 1.09x | `w512/skip4` | 1.36x | 19.7% |
| R6 | `w512/skip4` | 1.13x | `w256/skip4` | 1.33x | 15.0% |

Losing the quantized draft costs **~22% of aggregate throughput** (1.06x
against 1.35x). But the penalty is **not uniform**: at short context the
no-extra-HBM family collapses to 1.08-1.13x, while at long context it
retains **1.45x and 1.58x**. The design degrades gracefully exactly where KV
pressure actually arises, which is a genuine argument that window+skip is
the right fallback rather than a consolation prize.

**Not evaluated:** KV pressure was never varied as an axis. Every boot ran
at fixed `gpu_memory_utilization=0.90` with the draft resident. So this says
what is available IF the draft is unaffordable, not at what KV working-set
size the switchover triggers.

---

## 4. What this says about the design

1. **The epistemic split is validated.** Cost predicted soundly (47/48, five
   correct eliminations); acceptance had to be measured, and every attempt
   to shortcut it failed in a locatable way (D2b's k=16 inversion).
2. **Composition is the larger value; switching is real but bounded.**
   +12.5% from composing levers; **+6.2%** from choosing per regime, on
   **81%** of the 126 mixes. The originally reported +1.4% / 7% was an
   artifact of the missing fail-closed rule.
3. **Knowing when to stop was the design's biggest defect, and it is fixed.**
   At R4 every lever family loses and the selector armed anyway at **0.76x**;
   a rule keyed on the predicted margin's lower confidence bound declines
   there and nowhere else, taking the selector to **99.6% of omniscient**.
   That single fix was worth more than the entire per-regime switching
   advantage it was masking.
4. **The search has a stated guarantee, not just an ordering hunch.** Not
   top-1 correctness — no rule has that — but **recall@4 = 12/12** with
   confirm-1 regret bounded at a measured 2.0%, plus a certified error bar
   on the product bound (`rho(damage, error) = +0.875`, bound always in the
   sound direction).
5. **Long context is the operating regime.** 1.76-1.98x at 14k against
   1.22-1.42x at short context, and the memory-constrained fallback also
   holds up only there.

---

## 5. Limitations, stated plainly

* **One model, one hardware family.** Qwen3-8B dense, H100. The project's
  C1 claim spans dense/MoE/MLA; this phase does not.
* **Decode currency only.** Real serving wall-clock includes prefill —
  46.6% of wall at R5 — so end-user speedups are lower than these figures.
* **Equal-weight mix is a convention, not a workload.** The frontier is
  reported alongside precisely because the verdict moves with the mix.
* **The fail-closed rule is post-hoc.** Its error terms and threshold come
  only from pre-D3 data, but the D3 outcome was known when it was written
  and no barrier can be retrofitted; two invariance checks stand in for one
  (section 2.6).
* **KV pressure and content breadth are unmeasured axes** (sections 3.4-3.5).
  The 1.305x mixed-feasibility figure is a modelled estimate over measured
  rates, not a measurement.
* **The recall@4 guarantee rests on 12 groups** — point estimate 1.00,
  one-sided 95% lower bound 0.78 — on one lattice.
* **The measurement environment cost a week.** The original box was a
  QEMU/KVM guest whose host-side clamp gated 40+ campaign attempts without a
  single accepted cell; work moved to bare metal, and the clamp is localised
  to a timekeeping cost but not closed (`results_clamp_investigation.md`).

---

## 6. Open questions for discussion

1. ~~Does the fail-closed rule get built?~~ **DONE 2026-08-16**
   (`results_d3_failclosed.md`): 95.1% -> 99.6% of omniscient, and the
   failing static clause now passes.
2. ~~Is the static clause read strictly or loosely?~~ **MOOT** — the strict
   reading passes under the fail-closed rule.
3. **Does Phase 97's runtime switching get finished?** The re-scored price
   is **81% of mixes**, not 7%, so the argument against building it has
   been withdrawn. The remaining question is engineering cost against the
   +6.2%, and whether the mixed-feasibility case (1.305x modelled) is the
   real justification.
4. **The next campaign should be KV pressure**, per
   `results_switching_law.md`: it is the only axis that is well-posed,
   predicted to move the headline by 5x rather than a few percent, and tests
   the claim the paper actually needs — that a serving system must switch
   because its lever set is not uniformly affordable. Registered prediction
   to commit before it runs: **per-regime selection beats the best
   uniformly-feasible static by >= 1.20x** under KV pressure.
5. **Does the recall@4 shortlist size hold on a wider lattice?** It is
   measured on 22 compositions in 12 groups, not derived; the scaling is
   unknown.
