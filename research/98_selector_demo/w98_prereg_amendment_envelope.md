# Preregistration amendment 1 — the D1 prediction envelope

Date: 2026-08-13
Status: **DRAFT, awaiting approval.** Not in force. No held-out cell may be
booted under this rule until it is approved and hash-bound.

Amends: `w98_prereg.md` §D1. The frozen lattice, the held-out split, and the
`epsilon_arm = 0.015` elimination rule are **unchanged**.

## 1. What is wrong with the registered rule

D1 registers that "95% intervals cover the held-out composed configurations".
The implementation derives the interval from the fit residuals alone:

```python
envelope = max(|log(predicted / measured)|) over the 8 single-lever fit points
lo, hi   = point * exp(-envelope * 2.0), point * exp(+envelope * 2.0)
```

Round 1 produced envelopes of **0.007 to 0.030** — ±0.7% to ±3%. At R4 that is
±0.15 ms on a 22 ms quantity, against a per-boot `stdev_armed_step_s` of
~0.17 ms. **The interval was narrower than the instrument's own resolution.**

Nine of Round 1's twenty-one misses were 1.001–1.046x the band edge, with point
predictions within 2–5% of measurement. Those are not model failures; they are
the harness failing to state its own uncertainty. The remaining twelve were a
real composition effect (since root-caused to host starvation and fixed).

Three specific defects:

1. **No measurement term.** The envelope carries fit residuals only. Prediction
   error also includes the noise on the held-out measurement itself, which is
   never accounted for.
2. **`max` over 8 points is not a 95% interval.** It is an order statistic of a
   small sample, and it conflates model misfit with noise.
3. **`inflation = 2.0` is unjustified.** It is a magic constant, not a derived
   or registered conservatism.

## 2. The amended rule

For each regime `x`, the half-width of the D1 log interval becomes:

```text
envelope(x) = sqrt( fit_term(x)^2 + (Z * sigma_repro(x))^2 ) * INFLATION
Z         = 3
INFLATION = 2.0    (unchanged; see §4)
```

* **`fit_term(x)`** — unchanged: `max |log(pred/measured)|` over the single-lever
  fit points. Retained rather than replaced so the amendment changes exactly one
  thing, and because it is the conservative choice of the two.

* **`sigma_repro(x)`** — the **measured** log-scale boot-to-boot standard
  deviation of the step cost at regime `x`, from repeat boots of **anchor**
  configurations, defined in §3.

Both terms are recorded per regime in the result, so a reader can see which one
binds.

## 3. How `sigma_repro` is measured — and why it cannot be tuned

Constraints, all of which are part of the amendment:

* **Anchors are fit-set configurations only** — single-lever cells already in
  the fit. No held-out cell contributes to the envelope it is scored against.
  Two anchors: `target-matching/woff/skip0` (the unlevered baseline) and
  `w4a16-quantized/woff/skip0` (the quant single).
* **At least 3 repeat boots per anchor per regime**, and they must **bracket the
  campaign** — at least one before the first held-out boot and at least one
  after the last. Back-to-back repeats understate drift: the two whole-chain
  repeats in X6 were consecutive and gave CV 0.03%, while X5's interleaved
  repeats over ~30 minutes with load swinging 8.8→14.7 gave 1.70% on the same
  box.
* **`sigma_repro` is the maximum across anchors**, not the mean.
* **Measured on the runtime under test.** Reproducibility is a property of the
  configuration, not the box: piecewise measured 1.70% and whole-chain 0.03–0.15%
  on the same hardware, a 10–50x difference, because whole-chain takes the host
  out of the inner loop.
* **Registered before any held-out cell is booted**, and hash-bound with the
  authorization, exactly as the rest of the contract.

The rule is one-directional by construction: a noisier campaign yields a wider
envelope and an *easier* D1. §5 exists so that cannot be mistaken for a pass.

## 4. What is deliberately NOT changed

* **`INFLATION = 2.0` stays.** It is unjustified, but changing it in the same
  amendment that adds a measurement term would confound the two. It is
  registered as a standing conservatism for extrapolating from single-lever
  support to composed configurations, and left for a separate amendment if a
  principled value can be derived.
* **`fit_term` stays a max**, per §2.
* **The frozen lattice, held-out split, and `epsilon_arm`** are untouched.

## 5. Validity condition — a wide envelope is not a pass

If `Z * sigma_repro(x) > fit_term(x)` for a regime, that regime's D1 result is
reported as **NOT RESOLVABLE at this precision**, not as covered. A test that
passes only because the instrument is imprecise is not evidence for the model.

This makes the amendment cut both ways, which is the point of registering it
before the data.

## 6. What this would have done to Round 1 — illustrative only, not scored

Round 1 is a frozen record and is not rescored. Using X5's measured piecewise
reproducibility (CV 1.70% → `sigma_repro` ≈ 0.0169, `Z*sigma` ≈ 0.0506) against
the R4 fit term of 0.0068:

```text
envelope = sqrt(0.0068^2 + 0.0506^2) * 2.0 = 0.102   (±10.7%)
```

That band would have covered all nine near-band misses (1.001–1.046x) and none
of the twelve quant×skip8 misses (1.187–1.524x). The amended rule separates
instrument from model exactly where the manual reading of Round 1 did.

But `Z*sigma_repro` (0.0506) exceeds `fit_term` (0.0068), so under §5 Round 1's
piecewise campaign would be **not resolvable** — which is the honest verdict on
a campaign whose instrument was noisier than the effect it was testing.

## 7. Why the re-measurement makes D1 resolvable

On the whole-chain runtime, boot-to-boot CV is 0.03–0.15%
(`sigma_repro` ≈ 0.0003–0.0015, `Z*sigma` ≈ 0.0009–0.0045), against Round-1 fit
terms of 0.0068–0.0300. The fit term dominates, §5 is satisfied, and D1 becomes
resolvable.

So the runtime fix is not only a speedup: it is what makes the experiment
possible. That is worth stating in the paper — the composition claim could not
have been tested on the host-bound runtime at this lattice size, regardless of
how the envelope was defined.

Caveat: the whole-chain CV rests on **two consecutive** repeats. §3 requires
three bracketing the campaign, so the operative `sigma_repro` must be
re-measured under the amended protocol and may come out larger.

## 8. What approval commits us to

1. Boot the two anchors, 3 repeats each, bracketing the re-measurement campaign.
2. Re-measure the 15 lattice configurations on whole-chain + Marlin.
3. Fit, then score D1 over the 7 composed held-out cells with the amended
   envelope, reporting `fit_term` and `Z*sigma_repro` per regime and applying §5.
