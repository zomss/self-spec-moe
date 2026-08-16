# D2(b): the knapsack wins at k=4 and k=8, and inverts at k=16

Record of the D2(b) knapsack identity, 2026-08-16, h103 GPU 7. Per-layer
retention measured on content seed 4 (36 leave-one-out boots); sets
committed behind digest; confirmation measured on the disjoint seed 5 (18
boots). Numbers from `data/g98_d2b/d2b_result.json`.

## Verdict: FAILS as registered, by one control, at one count

**11 of 12 registered controls beaten; 1 lost.**

| count | knapsack tau | worst control beaten | result |
| --- | --- | --- | --- |
| 4 | **7.376** | 1.134 | beats all 4 |
| 8 | **6.423** | 1.071 | beats all 4 |
| 16 | **1.244** | 1.000 | **loses to `random_s98001` (1.970)** |

The registered claim is that the knapsack beats its count-matched controls
at *each* surviving count, so one loss fails it. But the failure is not
diffuse — it is one count, and the reason is precise.

## The knapsack's own objective predicts perfectly, until it doesn't

The knapsack maximises the product of per-layer retentions. Ranking every
arm by that product against measured tau:

| count | Spearman(predicted product, measured tau) |
| --- | --- |
| 4 | **+1.000** |
| 8 | **+1.000** |
| 16 | +0.771 |

At k=4 and k=8 the product model ranks all six arms **exactly right** —
perfect order, no inversions. Per-layer retention, measured by
leave-one-out, is a sufficient statistic for choosing skip sets at these
counts.

At k=16 it breaks where it matters. The bottom of the ranking is still
correct (every set containing layer 0 is correctly predicted terrible), but
the **top inverts**: the knapsack's set has by far the highest predicted
product (0.585, against 0.275 for the next) and measures **1.244**, worse
than both the frozen nested set (product 0.275 -> tau 2.080) and
`random_s98001` (product 0.197 -> tau 1.970).

So the product model **identifies bad sets at every count and stops
identifying the best set once the count is aggressive.** That is Phase 90's
non-additivity theorem reproduced on a new lever, a new model and a new box
— and, unlike Phase 90's control, it comes with the boundary located: the
model holds at 4 and 8 layers of 36, and has inverted by 16.

## k=16 is outside the lever's usable range entirely

Every k=16 arm collapses: measured tau spans **1.000 to 2.080**, against
6.4-7.4 at k=8 and k=4. Skipping 16 of 36 layers destroys the draft
regardless of which layers are chosen; the comparison at that count is
between degrees of ruin.

This is consistent with why skip16 exists. The lattice comment records it
was admitted for W98-R2 because "it is what makes the non-layer cost F
estimable" — a COST-identifiability choice, widening the keep range from
0.778-1.0 to 0.556. It was never claimed to be a good operating point, and
D2(b) now measures that it is not one for acceptance.

**A candidate mechanism, and why it is not asserted.** The knapsack's k=16
set is 15 contiguous layers (2-16), and contiguity correlates with the
collapse at k=16 (corr -0.681). But it fails as a general explanation: at
k=8 the knapsack has the LONGEST contiguous run of any arm (5) and the BEST
tau. So contiguity is not the mechanism, or not the only one, and the record
says so rather than over-reading three points.

## The dominant single factor is layer 0

Skipping layer 0 alone drops tau from 8.083 to **1.244** — retention 0.034,
where the next-worst layer retains 0.79. Every arm containing layer 0
collapses to tau 1.0-1.2 whatever else it contains. The knapsack's decisive
wins at k=4 and k=8 are partly this: `random_s98003` and both worst sets
include layer 0 and are destroyed by it.

But that is not the whole story, which the reference arm shows. Against
`random_s98001` at k=8 — a set that avoids layer 0 entirely — the knapsack
still wins 6.423 to 4.506. The selection is doing finer work than avoiding
one catastrophic layer.

## The frozen lattice, measured at last

Carried as a non-registered REFERENCE arm: the nested `skip4/8/16` sets the
cost campaigns actually booted.

| count | frozen tau | knapsack tau | knapsack advantage |
| --- | --- | --- | --- |
| 4 | 7.276 | 7.376 | +0.099 |
| 8 | 5.327 | 6.423 | **+1.096** |
| 16 | **2.080** | 1.244 | **-0.836** |

At k=4 the frozen set was already near-optimal — it differs from the
knapsack by one layer (7 against 9) and costs 0.1 tokens. At k=8 the
knapsack is meaningfully better, so the nested construction was leaving
about a token on the table. At k=16 the frozen set is BETTER than the
knapsack, for the same reason the knapsack fails there.

The nested sets were chosen before any per-layer measurement existed, and
they avoid layer 0 at every count. That was a good choice, and it is now
measured rather than assumed.

## Protocol

* **Non-scored retention probe.** Skip count 1 was admitted under `w98-d2`
  only, additively; `w98-lattice`, the scope every scored cost campaign
  booted under, keeps `{0,4,8,16}` unchanged. Route recorded 2026-08-16
  before any retention boot.
* **Disjoint data, per Phase 84.** Retention on seed 4, confirmation on seed
  5. The sets are never chosen using the content that judges them.
* **Commitment barrier.** Sets committed with a digest before any
  confirmation existed; `commit_sets` refuses afterwards.
* **Frozen controls.** Random-set seeds 98001-3 were fixed in
  `data/prereg/w98_d2_controls.json` long before this campaign.
