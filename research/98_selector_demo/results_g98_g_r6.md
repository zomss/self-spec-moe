# The R6 misrank: diagnosed, fixed, and not worth deploying

G98-G's selector missed at R6 by 12%, picking `w4a16/w256/skip8` where
`w4a16/woff/skip4` won. This localises the cause, fixes it, and then shows
why the fix should not ship as a global rule — which is the part worth
keeping.

## 1. It is not the cost model

Backing realized cost out of the measured run (`D_real = tau/rate - verify`)
against the fitted prediction, for every R6 candidate:

| cell | `D_hat` ms | `D_real` ms | error |
| --- | --- | --- | --- |
| `w1024/skip0` | 24.68 | 30.14 | −18.1% |
| `w1024/skip4` | 22.20 | 27.19 | −18.3% |
| `w1024/skip8` | 19.73 | 24.45 | −19.3% |
| `w128/skip8` | 18.22 | 23.13 | −21.2% |
| `w256/skip4` | 21.36 | 26.60 | −19.7% |
| `w256/skip8` | 18.99 | 23.84 | −20.3% |
| `w512/skip4` | 22.20 | 27.01 | −17.8% |
| `woff/skip0` | 24.25 | 29.48 | −17.7% |
| `woff/skip4` | 21.83 | 26.78 | −18.5% |

The model under-predicts everything by 17.7–21.2% — a **3.5-point spread**
across a 20-point bias. A near-uniform multiplicative error in `D` barely
moves a ranking, and substituting `D_real` while keeping transferred
acceptance still leaves the wrong cell on top. **The cost half is not the
defect.**

## 2. It is the acceptance transfer, and the error tracks skip depth

Transferred acceptance (h104, content seeds 4–5) against realized
acceptance on the evaluation content (seeds 2–3):

| depth | cells | optimism of transferred tau |
| --- | --- | --- |
| skip0 | 2 | +3.4%, +6.3% |
| skip4 | 4 | +5.0% … +9.6% |
| skip8 | 3 | **+16.4% … +17.1%** |

The deeper the skip, the worse acceptance generalises to unseen content — so
the selector over-rated the deep-skip cell and picked it. Substituting
realized acceptance puts the measured winner on top immediately.

Two corrections were tried and rejected before the one that worked:

* **A confidence bound from seed spread.** G98-D's own seed-4-vs-5 spread
  does rise with depth (median 0.59 / 0.85 / 1.20% for skip0/4/8), so the
  direction is right — but it is an order of magnitude too small to close a
  16% gap, and an LCB built from it does not change the pick.
* **Re-measuring acceptance on this box.** Acceptance is deterministic given
  checkpoints and prompts (G98-F measured bit-identical accept patterns
  across boxes), so re-running seeds 4–5 here reproduces h104's numbers
  exactly. The gap is content generalisation, not a box effect.

## 3. The fix: measure acceptance on the deployment content

This is the design's own answer — acceptance is the half that must be
measured — priced concretely by booting each candidate briefly on the
evaluation content and reading tau from it.

**Burn-in length matters, for a mechanical reason.** With a fixed token
budget, tau is inferred from an integer step count, so its resolution is
about `tau / steps`:

| burn-in | resolution | outcome |
| --- | --- | --- |
| 64 tokens | ~5.3% | **fails** — five candidates collide on tau = 3.316; pick is noise |
| 256 tokens | ~1.3% | **fixes it** — pick becomes `woff/skip4`, regret 12.0% → **0.0%** |

Across all six regimes, with 256 tokens:

| regime | offline pick regret | with burn-in |
| --- | --- | --- |
| R1 | 0.00% | 1.46% |
| R4 | 1.59% | **0.00%** |
| R5 | 0.00% | 1.14% |
| R5cot | 0.59% | 0.59% |
| R6 | **11.97%** | **0.00%** |
| R8 | 0.00% | 0.00% |
| **mean** | **2.36%** | **0.53%** |

Mean per-regime regret falls **4.4x**. The big miss is gone; two regimes
pick up ~1.2–1.5% where candidates were near-tied and the burn-in's noise
decides between them. That is the bias–variance trade stated plainly:
transferred acceptance is low-variance and biased, burn-in acceptance is
unbiased and noisy, and each wins where its weakness is smaller.

## 4. Why it should not ship as a global rule

The equal-weight mix is over TOKENS, so time share goes as `1 / rate`:

| regime | OFF rate | share of time |
| --- | --- | --- |
| **R1** | 106.8 tok/s | **59.3%** |
| R5cot | 524.6 | 12.1% |
| R5 | 525.5 | 12.0% |
| R4 | 608.9 | 10.4% |
| R8 | 1573.5 | 4.0% |
| **R6** | 2930.9 | **2.2%** |

R6 carries **2.2%** of the time. Its 12% regret is therefore worth about
0.26% of end-to-end throughput, while the burn-in's new 1.46% regret at R1
costs 1.46% x 59.3% ≈ 0.87%. Measured aggregate:

| selector | aggregate | of best-in-set |
| --- | --- | --- |
| offline (transferred tau) | **584.1** tok/s | 0.9947 |
| + 256-token burn-in | 581.1 tok/s | 0.9896 |
| best-in-set (oracle) | 587.2 tok/s | — |

**−0.51% net.** A guarded variant (use burn-in only where it disagrees with
the transfer by >5%) was also tried and lands identically at 581.1, because
the disagreement is large at R1 and R5 too — the guard does not separate
bias from noise.

So the correct reading of the original 12% miss is that **it was expensive
to look at and cheap to own**. Per-regime regret is the wrong currency for
deciding where to spend measurement; time-weighted contribution is the right
one, and it says the regimes worth measuring acceptance for are the SLOW
ones — low batch, long context — not the fast one where the misrank happened.

## 5. What ships

* The diagnosis, which is reusable: cost errors that are near-uniform across
  candidates do not misrank; acceptance errors that scale with lever
  aggression do.
* `probe_w98_r6_burnin.py`, parameterised by burn-in length and regime set,
  with the 64-token failure kept in the record as the resolution floor.
* **No change to the selector.** The burn-in correction is not enabled: it
  improves per-regime regret 4.4x and end-to-end throughput not at all.
  Enabling it would need the burn-in budget spent per regime in proportion
  to time share — long enough at R1 to beat its own noise — which is a
  design change this run does not license.

## 6. Limits

* Section 3's evaluation reuses the grid that revealed the miss, so the
  256-token result is **in-sample** for the choice of burn-in length; only
  the two lengths actually tried are reported, and the 64-token failure is
  reported alongside precisely because it was not a success worth hiding.
* One box, one mix. The time-share argument depends on the equal-weight
  token mix; a mix dominated by short-prompt high-batch traffic would move
  R6's weight up and could flip the conclusion.
* The burn-in shares content with the scored run. A deployment burn-in sees
  the same traffic it then serves, which is the intended analogue, but it
  means this does not test generalisation to yet-unseen content.
