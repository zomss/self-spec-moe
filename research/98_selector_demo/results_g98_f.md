# G98-F — the fail-closed rule, and what validating it found

The rule was requested as the phase's highest-value fix: D3 scored the
selector at 95.1% of omniscient with essentially the whole gap in one cell,
R4, where speculation measured 522.3 against OFF's 684.9 and the selector
armed anyway. Building it was straightforward. **Validating it refuted the
defect it was built to fix.**

Both results are reported. The rule is sound and is kept; the 24% R4
regression it targets does not reproduce.

## 1. The rule (`w98_failclosed.py`, `w98_prereg_failclosed.md`)

```text
margin(R)    = predicted[best armed][R] / predicted[off][R]
threshold(R) = max(EPSILON_ARM, envelope(R))
arm(R)  iff  margin(R) - 1 > threshold(R)
```

Arm only when the predicted advantage exceeds what the cost model can
resolve. **No free parameter**: `EPSILON_ARM = 0.015` is the Round-1 arming
rent and `envelope(R)` is the D1' combined envelope from the closed G98-C
fit. Both predate the rule. A regime with no fitted envelope raises rather
than arms.

Committed before scoring (`data/g98_f/commitment.json`, digest
`bea15a46…`): **firing set {R4}**, every armed regime clearing its
threshold by more than 3x.

| regime | margin | threshold | decision |
| --- | --- | --- | --- |
| R1 | 1.335 | 0.0254 | arm |
| **R4** | **1.026** | **0.0724** | **park** |
| R5 | 1.632 | 0.1362 | arm |
| R5cot | 1.805 | 0.0956 | arm |
| R6 | 1.228 | 0.0323 | arm |
| R8 | 1.180 | 0.0174 | arm |

## 2. In-sample: it does what it was asked to do

Applied to the scored D3 grid, in D3's own time-weighted currency
(`score_w98_failclosed.py`, `data/g98_f/result.json`):

| quantity | baseline | fail-closed | |
| --- | --- | --- | --- |
| share of omniscient (corrected) | 0.951 | **0.996** | |
| share of omniscient (legacy) | 0.948 | **0.996** | |
| over static OFF (corrected) | 1.371x | **1.436x** | |
| over the best single static | 0.94x | **1.062x** | clause flips to PASS |

R4 is the only pick that changes, 522.3 -> 684.9. Both instrument arms agree.

**This is in-sample by construction** — the rule was written after seeing R4
fail in this grid — and is reported as the consequence of a pre-committed
rule, not as evidence.

## 3. Out-of-sample: the premise fails

`probe_w98_failclosed_validate.py`, 12 boots, two lanes (GPU 0 and GPU 1),
interleaved with arm order reversed between rounds, both regimes measured
inside each boot.

| regime | armed | OFF | armed/OFF | h103 (D3) | rule says |
| --- | --- | --- | --- | --- | --- |
| **R4** | **726.8** | 643.5 | **1.129** | **0.763** | park |
| R1 | 165.0 | 114.8 | 1.437 | 1.423 | arm |

GPU 1 replicate: R4 **1.131**, R1 **1.428**. Verdict
`PREMISE1_REFUTED_ARMED_WINS_AT_R4`.

The control is what makes this readable. **R1 reproduces h103 to 1%**
(1.437 against 1.423), so the two boxes measure the same thing in the same
way — and at R4 they disagree by 48%.

## 4. Which measurement is the outlier?

Six independent lines, all pointing at h103's R4 armed column:

1. **This box is uniformly slower except there.** R1-off 0.916, R4-off
   0.940, R1-armed 0.926 of h103 — then R4-armed **1.391**.
2. **Correcting for that box factor**, h103's R4-armed "should" have read
   ~786 tok/s; it read 522.
3. **Work is bit-identical**: 158 steps, 157 armed, 5084 committed tokens at
   R4 on both boxes; 133/132/639 at R1. Same acceptance, same schedule —
   only time differs.
4. **The cost model, fit on a third box (h104), agrees with this box.**
   Measured/predicted armed-over-OFF is 0.99–1.15 for every regime on h103
   except R4 at **0.743**.
5. **Quantization inverts only at R4 on h103** — it helps at every other
   regime (1.05–1.26x) and hurts there (0.823x). A lever that cuts draft
   weight reads from 16.4 GB to 6.1 GB cannot make the draft slower.
6. **The armed R4 step is the grid's most bandwidth-hungry** (five forwards
   per step, window off, 8k context, batch 8, separate quantized weights) —
   the step most exposed to memory-bandwidth contention, which the phase has
   already recorded moving h104 boots by +76% under same-node co-activity.

The parsimonious reading: **h103's R4 armed cells were measured under
contention**, and the D3 grid has no replicate that could have caught it —
each cell was booted once per arm, with no timing gate.

## 5. What this means

* **The R4 regression is not established.** The phase's headline defect —
  "the selector's one large miss", "the single highest-value fix (+24%)" —
  rests on unreplicated boots that an independent box contradicts by 48%.
* **On this box the rule would COST 11%** at R4 (parking 726.8 to take
  643.5), not gain 24%.
* **The rule's principle survives, and is arguably strengthened.** R4's
  armed-over-OFF margin spans 0.76 to 1.13 across boxes: genuinely
  unresolvable, which is exactly what its 0.072 envelope — the second widest
  in the map — was saying. Declining to arm on a margin you cannot measure
  remains correct; what is now unclear is what declining is worth.
* **Deployment is therefore held**, not cancelled. The rule ships as code,
  tested and committed, with its firing set committed and its effect
  documented — but it is not enabled at R4 until the discrepancy is closed.

## 6. What closes it

**Replicate h103's R4 column.** Six boots — OFF, `w4a16/woff/skip0`,
`target-matching/woff/skip0`, twice each, interleaved — on h103, measuring
R4 and R1. R1 is the built-in control: it must land near 1.42.

* If h103 reproduces ~0.76 with R1 clean, the discrepancy is a real property
  of that machine, the rule is right for h103, and the phase gains a
  box-dependence result it does not currently have.
* If h103 now reads ~1.1, **D3's R4 row is wrong**, the 24% regression
  disappears, the selector needs no fix at R4, and D3's static clause
  verdict must be re-derived from a corrected grid.

Until then the honest statement of D3 is: 95.1% of omniscient, with the one
large miss **unconfirmed**.
