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

## 4. The sixth hypothesis lands: acceptance was measured on filler

Every acceptance number this project owns was measured under `ignore_eos`.
Measured both ways on the same arms, same cell, same batch:

| protocol | `w1024/skip4` | `w1024/skip8` | winner |
| --- | --- | --- | --- |
| **`ignore_eos`** (all curves) | tau **4.407** | 3.623 | skip4 by +21.6% |
| **natural EOS** (real content) | tau 2.278 | **2.392** | **skip8 by +5.0%** |

**The ordering reverses.** Section 35 established that filler past the
natural stopping point is trivially predictable; this shows the inflation is
**arm-dependent** -- a less-damaged draft has more headroom toward the
ceiling on easy text, so filler flatters `skip4` and the advantage vanishes
on real content.

Two consequences:

* **It explains the whole chain.** `w1024/skip8` wins real rollouts because
  it genuinely accepts more there. Our table says the opposite, which is why
  section 48's model ranked it 5th against a measured 1st, why E1's schedule
  never selected a skip8 arm, and why the equal-work band tables disagree
  with section 45.
* **Absolute acceptance is roughly half what we believed**: tau ~2.3 on real
  content against 3.6-4.4 on filler. Every predicted speedup built on those
  curves is optimistic.

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

**Damaged (built on filler acceptance):** every u-resolved curve (s35, the
80-boot lattice campaign), every prediction map, section 48's shortlist
scoring and confirmation-budget curve, section 49's tau(K) analysis, and
E1/E1c's schedules and margins.

**Not damaged (natural EOS throughput):** section 45's 14-point lattice and
its winners -- replicated here to 0.21% -- section 30's stock baselines,
section 32-33's instrument corrections, and the cost model, which is fitted
on draft-chain timings and carries no acceptance term.

## 7. E1's conclusion, restated

E1 reported +0.4% for intra-rollout switching and fired the registered
under-3% criterion. That number is **not trustworthy**: it was computed over
a candidate set the model ranked wrongly because the acceptance input was
contaminated, and the arm that wins real rollouts was never in any schedule
the optimizer produced. E1 is **suspended**, not concluded, pending
re-measured acceptance.

## 8. Next

Re-measure acceptance under **natural EOS** with per-position profiles. The
probe hardcodes `ignore_eos=True` for a reason section 35 records -- it
guarantees the deep u-buckets populate -- and that reason is now in direct
conflict with correctness. Under natural EOS the deep buckets fill only from
the long tail of the length distribution, so they will be thinner; that is
the honest cost of measuring the right thing.
