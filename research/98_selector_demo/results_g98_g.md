# G98-G — the whole selector, searched and evaluated on one box

The phase built its selector across three machines: cost fitted on h104,
acceptance measured on h104, grid scored on h103. This runs the search and
its evaluation end to end on a single box — the KVM guest — so the design is
judged without cross-box transfer, and so the search is forced to work from
cost profiles that genuinely differ (this box's quantized draft chain is 27%
slower than h104's while its unquantized chain is only 6% slower).

**Result: the search reaches 99.5% of the best configuration it considered,
at 1.54x over no speculation, from 17 gated cost boots and zero acceptance
measurements of its own.**

## 1. What ran

| stage | boots | what it produced |
| --- | --- | --- |
| cost | 17 (gated, ratcheted over 11 passes) | draft-chain and verify cost per regime |
| fit | — | five-parameter model per regime, predicting all 31 cells |
| commit | — | prediction map frozen behind digest `eafb7b1f` |
| confirm | 20 (10 cells x 2 rounds, interleaved) | measured decode throughput, 6 regimes |

Acceptance was **not** measured here. It is taken from h104's G98-D, which
is sound because acceptance is a property of checkpoints, prompts and arming
schedule rather than of the machine — and the confirmation re-derives
realized acceptance for every cell so the transfer is checked, not assumed
(section 4).

## 2. The search picks the same configurations as h103's

Fitted only from this box's own profiles, the search reproduces h103's picks
in **5 of 6 regimes**:

| regime | this box picks | h103 picked | same? |
| --- | --- | --- | --- |
| R1 | `w4a16/woff/skip4` | `w4a16/woff/skip4` | yes |
| R4 | `w4a16/woff/skip0` | `w4a16/woff/skip0` | yes |
| R5 | `w4a16/w512/skip4` | `w4a16/w512/skip4` | yes |
| R5cot | `w4a16/w512/skip4` | `w4a16/w512/skip4` | yes |
| R6 | `w4a16/w256/skip8` | `w4a16/woff/skip8` | no |
| R8 | `w4a16/woff/skip4` | `w4a16/woff/skip4` | yes |

The cost surface differs materially between the boxes and the *choice*
mostly does not. Quantization is on in every pick, as everywhere else in
this phase.

## 3. What the measurement says

10 cells, 6 regimes, 2 interleaved rounds with order reversed, 20 boots:

| regime | selector pick | tok/s | vs OFF | best in set | regret |
| --- | --- | --- | --- | --- | --- |
| R1 | `w4a16/woff/skip4` | 164.0 | 1.536x | (same) | **0.00%** |
| R4 | `w4a16/woff/skip0` | 728.4 | **1.196x** | `w4a16/woff/skip4` | 1.59% |
| R5 | `w4a16/w512/skip4` | 883.9 | 1.682x | (same) | **0.00%** |
| R5cot | `w4a16/w512/skip4` | 1002.2 | 1.910x | `w4a16/w1024/skip4` | 0.59% |
| R6 | `w4a16/w256/skip8` | 3863.4 | 1.318x | `w4a16/woff/skip4` | **11.97%** |
| R8 | `w4a16/woff/skip4` | 2422.8 | 1.540x | (same) | **0.00%** |

Equal-weight, time-weighted aggregate (D3's currency):

* selector **584.1** tok/s, best-in-set **587.2**, OFF **379.9**
* **selector / best-in-set = 0.995**
* **selector / OFF = 1.538x**

It picks the exact winner in 3 of 6 regimes and lands within 1.6% in 5 of 6.

**R6 is the one real miss, at 12%.** It is the same regime where h103's
selector also missed its omniscient pick, so this is a cost-model weakness at
batch 32 rather than a box artifact: both boxes' fits misrank the window
there, in opposite directions.

## 4. The acceptance transfer holds

Realized acceptance re-derived from all 54 measured (cell, regime) pairs,
against the transferred h104 values:

* median ratio **0.971**, range 0.854–1.098.

The transferred numbers run ~3% optimistic. That is expected and not an
error: G98-D measured acceptance on content seeds 4–5 while this grid runs
seeds 2–3, so the selector is predicting **unseen content** by design. A ~3%
optimism applied uniformly across candidates barely moves a ranking, which
is why the picks survive it.

## 5. R4, for the third time

This box measures R4 armed at **1.196x over OFF**, over the full candidate
set. With G98-F's two lanes (1.129x, 1.131x) that is three independent
confirmations against h103's 0.763x, and it is now the *aggregate* R4
result rather than a two-cell comparison.

The consequence for the fail-closed rule is worth stating plainly. Evaluated
on **this box's own** map and fits, the rule still fires at R4 — the margin
is 1.033 against a threshold of 0.190, since the wider fit here makes the
envelope wider too — and parking there would cost:

* selector as measured: **584.1** tok/s
* selector with the rule: **569.2** tok/s (**-2.6%**)

So the rule remains correctly *motivated* (R4's margin genuinely sits inside
the model's error bar on both boxes) and empirically *wrong to enable* on
every box that has actually been measured. The pending h103 replication
(`results_g98_f.md` section 6) is what decides whether R4 is a machine
property or a bad measurement; nothing here should be read as endorsing the
rule's deployment.

## 6. Scope and limits

* **This is a candidate-set claim, not a share-of-omniscient claim.** 10
  cells, not 31: `selector / best-in-set` is a weaker bar than D3's
  `selector / omniscient`, and the two numbers must not be compared
  directly. The set does include h103's omniscient picks, so it is not
  trivially easy, but a cell outside the set could still beat the pick.
* **Why not the full grid**: a 31-cell grid compares cells across boots and
  this box changes state between boots. Grid boots carry no profiler, so the
  proven draft-chain gate cannot be applied, and a step-time proxy fails on
  h103's own clean data (3 of 31 cells non-monotonic; minimum spread 0.017
  overlapping the clamped range). Measuring it anyway would have produced a
  scored-looking number with no way to separate contaminated cells.
* **One lane, not two.** A co-tenant occupied GPUs 1–7 mid-run (~25 GB and
  27–74% utilisation each), so the lane-B replicate could not allocate and
  its boots failed. The two interleaved rounds on lane A remain, with order
  reversed, but the cross-lane replicate is missing.
* **The cost fits are looser here** than h104's — fit terms 0.022–0.132
  against 0.007–0.068 — which is expected on a box whose state moves, and is
  why the fail-closed thresholds computed here are so wide.
* Decode currency only, one model, K=4.
