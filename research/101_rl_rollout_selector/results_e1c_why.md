# E1c-E1g — why switching looked dead, and the protocol defect that explains it

Date: 2026-08-22. Investigation of E1's +0.4% result. Six hypotheses tested;
five refuted; the sixth is a measurement-protocol defect upstream of nearly
every acceptance number this project owns.

## 1. The decomposition (E1c, model-space, free)

**Half of every step is arm-independent.** Verify plus the draft's fixed
floor `F`:

| operating point | floor | share of step |
| --- | --- | --- |
| u=512, B=32 | 24.6 of 45.5 ms | **54.1%** |
| u=16K, B=11 | 19.3 of 37.1 ms | 52.0% |

**On the half the levers touch, the survivors sit on a flat frontier.** At
u=4096, top-6 arms: step-cost spread **18.9%**, acceptance spread **17.2%**,
ms/token spread **5.7%**. Cost and acceptance trade at nearly the break-even
rate, which is what "already selected onto the efficiency frontier" looks
like.

**Margin x dwell is the whole story.**

| band | time share | local margin | contribution |
| --- | --- | --- | --- |
| 0-1K | 3.2% | **5.39%** | 0.175% |
| 1-4K | 10.8% | 0.52% | 0.057% |
| 4-13K | 29.5% | 0.00% | 0.000% |
| 13-33K | 56.5% | 0.31% | 0.173% |

The early phase has a real margin and 3% of the time; the bulk of the
rollout sits where the static is locally optimal.

## 2. The crossing is real and measured (E1d)

LO b32, equal work, ms/token:

| arm | first 1K | first 4K |
| --- | --- | --- |
| `woff/skip4` | **best** | +12.49% |
| `w512/skip4` | +3.75% | **best** |

The ordering inverts between 1K and 4K exactly as the crossover predicted, so
the skip-early/window-later hypothesis is measured, not merely modelled. The
switching arithmetic also checks out: the eventual static loses 3.75% over
~2.7% of the rollout's time, contributing ~0.1%.

**A KV wall appeared here.** 32 requests x 16.5K tokens needs **78 GB** of KV
against ~50 GB free after weights and the 6.1 GB quantized draft; KV
exhaustion triggers preemption and recompute, and the registry then refuses
the 7,336-token prefill chunk scheduled under a width-1 action. At this model
and box, `B x G` is capped near 20 GB of KV: B=32 to ~4K tokens, or 16K
tokens at B=8. TP=2 is the minimum that reaches realistic rollout shape.

## 3. Five refuted explanations for the skip8 anomaly

`w1024/skip8` measures **1.3120 ms/token — best of twenty** in the section-45
natural-EOS LO b8 grid, and places **third** at equal work. Tested:

| hypothesis | verdict |
| --- | --- |
| token-count difference | refuted: 127,645 vs 124,845 (~2%) |
| drain shape / stragglers | refuted: efficiency 47-52%, uncorrelated with rank |
| skip pays off at low batch | refuted (E1e): skip8 is 3.7-12.9% SLOWER at every batch 1-16, no crossover |
| the cost/acceptance trade | refuted: -17.8% acceptance for -8.1% draft cost, a 2:1 loss |
| an unreplicated outlier | refuted: replicate reproduces to **0.21%**, identical token counts |

## 4. The sixth hypothesis: RETRACTED — it was an arithmetic error

**This section first reported that acceptance ordering REVERSES between
`ignore_eos` and natural EOS.** It does not. The finding was my own error and
is retracted in full.

The error: I computed acceptance as `E_committed / armed_steps / 8`, dividing
by the NOMINAL batch. Under natural EOS the batch drains, so the active count
is `H_target_steps`, and the two arms drain differently — mean active **4.20**
for `w1024/skip4` against **4.75** for `w1024/skip8`. That difference alone
manufactured the apparent reversal.

| | skip4 | skip8 | ordering |
| --- | --- | --- | --- |
| wrong figure (divided by 8) | 2.278 | 2.392 | skip8 ahead — **artifact** |
| **corrected (divided by active)** | **4.344** | 4.029 | **skip4 ahead by 7.8%** |
| `ignore_eos` curves | 4.407 | 3.623 | skip4 ahead by 21.6% |

The E2 re-measurement agrees independently: per-bucket filler inflation in
bucket 3 (u > 8192, where `ignore_eos` manufactures text) is **+0.1% to
+2.6%**, not a sign flip.

**What is true:** `ignore_eos` inflation is real and mildly arm-dependent —
it exaggerates `skip4`'s advantage over `skip8` from a real 7.8% to a
measured 21.6%. Section 35's original characterisation stands; the escalation
to "the ordering reverses" was wrong.

Two signals should have caught this sooner. A tau of 2.3 at K=4 is
implausibly low on its face. And E2's buckets 0-2 returned **bit-identical**
to the filler campaign, which is CORRECT — greedy generation is deterministic
until a request would stop, and LO's shortest is ~4,600 tokens, so nothing
can differ below u=3072. I read that as a bug rather than as evidence that
the aggregate was the outlier.

### The clue the correction produced

The arms **drain differently**: mean active 4.20 (skip4) against 4.75
(skip8), so `skip8` keeps ~13% more requests alive per step over the run.
Per-step measurements at fixed batch cannot see that, and it is the first
mechanism consistent with every observation — including that `w1024/skip8`
wins whole rollouts while losing every fixed-batch comparison. It is a
hypothesis, not a finding: why the arms drain differently is unexplained,
and the length totals are close (127,645 vs 124,845).

## 5. The knapsack set does not transfer either

LO b8 natural EOS, skip-8 layer identity:

| arm | nested set | knapsack-8 | change |
| --- | --- | --- | --- |
| `w1024/skip8` | 1.6595 | 1.4486 | **-12.7%** |
| `w512/skip8` | 1.6266 | 1.5468 | -4.9% |
| `woff/skip8` | 1.2936 | 1.0765 | -16.8% |

D2(b) measured a knapsack-chosen k=8 set beating the nested set on its own
protocol; it loses by 5-17% here. D2(b) selected on R-grid content at KMAX=8
**under `ignore_eos`** -- the same filler signal section 4 just invalidated
-- so the two negatives are plausibly one defect. The nested sets stay.

## 6. What is and is not damaged

With section 4 retracted, the damage assessment is much narrower than first
written:

**Mildly affected:** acceptance curves carry a few percent of `ignore_eos`
inflation, arm-dependent, which exaggerates gaps between arms of different
skip depth. Predictions built on them are correspondingly optimistic about
lightly-damaged arms. This is a calibration bias, not an inversion.

**Not affected:** section 45's 14-point lattice and winners (replicated to
0.21%), section 30's stock baselines, sections 32-33's instrument
corrections, the cost model, and section 48's ranking machinery — whose
known weakness on composed deep-skip arms is a separate, still-unexplained
defect.

## 7. E1's conclusion, restated

E1 reported +0.4% for intra-rollout switching and fired the registered
under-3% criterion. **The retraction in section 4 removes the reason to
suspend it**, but one concern survives on independent evidence: section 48
measured the model ranking `w1024/skip8` **5th where measurement puts it
1st**, and E1's optimizer never selected any skip8 arm in any schedule. So
E1's candidate set demonstrably excluded the arm that wins real rollouts, for
reasons that predate this investigation and are not about `ignore_eos`.

E1's +0.4% therefore stands as **a lower bound over a candidate set known to
be missing the measured winner**, not as a refutation of intra-rollout
switching. Re-running it once the skip8 anomaly is explained is the honest
close.

## 8. Next

Re-measure acceptance under **natural EOS** with per-position profiles. The
probe hardcodes `ignore_eos=True` for a reason section 35 records -- it
guarantees the deep u-buckets populate -- and that reason is now in direct
conflict with correctness. Under natural EOS the deep buckets fill only from
the long tail of the length distribution, so they will be thinner; that is
the honest cost of measuring the right thing.
