# Preregistration amendment 1 — the D1 prediction envelope

Date: 2026-08-13
Status: **APPROVED 2026-08-13** (Z=2, INFLATION retained). In force for the
re-measurement campaign once §9's blocker is resolved. Must be hash-bound with
the campaign authorization before any held-out cell is booted.

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
Z         = 2      (matches the registered "95% intervals" wording)
INFLATION = 2.0    (unchanged; see §4)
```

`Z = 2` was chosen over a more conservative 3 precisely because D1 registers
"95% intervals"; raising Z would have been a silent deviation from the
registered claim in the direction of making D1 easier.

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

* **Anchors must not be held-out cells**, so nothing that is scored contributes
  to the envelope it is scored against. Fit-set cells satisfy this; so does any
  lattice cell that is neither fitted nor held out.
* **One anchor per runtime class** (see §9), since reproducibility is a property
  of the runtime, not the box. `target-matching/woff/skip0` anchors the
  piecewise class; `target-matching/w256/skip0` anchors the whole-chain class.
  Both are fit-set cells.
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
reproducibility (CV 1.70% → `sigma_repro` ≈ 0.0169, `Z*sigma` = 0.0337 at Z=2)
against the R4 fit term of 0.0068:

```text
envelope = sqrt(0.0068^2 + 0.0337^2) * 2.0 = 0.0688   (±7.1%)
```

That band would have covered all nine near-band misses (1.001–1.046x) and none
of the twelve quant×skip8 misses (1.187–1.524x). The amended rule separates
instrument from model exactly where the manual reading of Round 1 did.

But `Z*sigma_repro` (0.0337) exceeds `fit_term` (0.0068), so under §5 Round 1's
piecewise campaign would be **not resolvable** — which is the honest verdict on
a campaign whose instrument was noisier than the effect it was testing.

## 7. Why the re-measurement makes D1 resolvable

On the whole-chain runtime, boot-to-boot CV is 0.03–0.15%
(`sigma_repro` ≈ 0.0003–0.0015, `Z*sigma` ≈ 0.0006–0.0030), against Round-1 fit
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

## 9. BLOCKER — the frozen lattice cannot support a whole-chain fit

Recorded here because it was discovered applying this amendment, and it
supersedes §8's plan.

FULLCG requires `KV_WINDOW > 0`, so `window: off` cells cannot run the
whole-chain runtime. Mapping the frozen fit set onto runtime classes:

| single-lever profile | axis it samples | runtime available |
| --- | --- | --- |
| target-matching/woff/skip0 | baseline | piecewise only |
| target-matching/w{128,256,512,1024}/skip0 | **window** | whole-chain |
| target-matching/woff/skip4 | **skip** | piecewise only |
| target-matching/woff/skip8 | **skip** | piecewise only |
| w4a16-quantized/woff/skip0 | **quant** | piecewise only |

**The skip and quant axes are sampled only at `woff`.** Consequences:

* A **whole-chain-only fit is unidentifiable** — it would have window
  information and nothing else, so `kappa_w` and the `keep_frac` coefficient
  cannot be estimated.
* A **mixed fit is invalid** — "unwindowed" and "uncaptured" are the same
  indicator, so the 10-13 ms/step whole-chain saving is perfectly collinear with
  the window axis. The fit would attribute the runtime fix to the window lever
  and then mispredict every composed cell.

This is a lattice-level problem, not an envelope problem, and it needs its own
decision before any re-measurement:

* **Option A — re-measure on piecewise.** Valid under the frozen prereg, needs
  no lattice change, and yields a correctly-enveloped D1. But it scores a
  runtime we have superseded, and by §5 it is likely NOT RESOLVABLE (piecewise
  `Z*sigma_repro` 0.0337 vs fit terms 0.0068-0.0300).
* **Option B — re-parameterise the lattice for the whole-chain class.** Give
  that class its own baseline (`target-matching/w256/skip0`) and its own
  single-lever profiles, so skip and quant are each sampled windowed
  (`target-matching/w256/skip4`, `w4a16-quantized/w256/skip0`). Roughly the same
  boot count. This changes the frozen fit set, so it is a **new round with a new
  preregistration**, not a re-measurement.

Recommendation: **B**, because A measures a runtime we do not intend to use and
is probably not resolvable anyway. But B is a new preregistration and that is
the user's call, not a drafting decision.
