# G98-B — Round 1 result (v5)

Date: 2026-08-12

`D` is the profiler's `draft_chain` region, not the engine step. D1 is scored
over the **7 genuinely composed** held-out cells; the 8th registered cell is
excluded with its reason recorded, and the frozen split is not rewritten.

**Coverage: 21/42.** Every one of the 21 misses is in the same direction —
measured above the band, never below. The misses separate into two groups that
mean different things, and only one of them is a composition effect.

## What changed from v4, and why

v4 measured the whole engine step and called it `D`. The step contains the
target verify (6.2 ms at R1) and ~3.7 ms of scheduler/sampler residual that is
flat across every configuration and every context length measured. The factored
model multiplies its whole expression by `keep_frac`, so those constants were
being scaled down as layers were skipped — an error that grows with skip depth,
which is exactly the shape v4 reported (1.05× at skip4, 1.44× at skip8).

Correcting an earlier statement in the v5 commit message: the constant is **not**
mostly the verify. Verify is ~15% of the step. The fitted `c0` was 61–81% of the
step because it was absorbing the draft chain's own launch-bound floor as well.
That floor *does* belong inside `keep_frac` (skipping layers removes launches);
the verify and the residual do not. Measuring `draft_chain` removes only the
~25% that never scaled.

## The two failure modes

| | cells | miss ratio | reading |
| --- | --- | --- | --- |
| near-band | 9 | 1.001–1.046× | band narrower than the harness resolves |
| structural | 12 | 1.187–1.524× | one real composition effect |

### Near-band: the envelope is too tight to be a test

Fitted log envelopes are 0.007–0.030, i.e. **±0.7% to ±3%**. At R4 the band on a
22 ms quantity is ±0.15 ms. The point predictions for these cells land within
2–5% of measurement, which is a good factored fit; they are scored as misses
because the interval demands a precision the measurement does not claim. The
per-boot `stdev_armed_step_s` is ~0.17 ms, comparable to the whole band.

This is a harness defect, not evidence about composition.

### Structural: quantized × skip8 × windowed costs ~48% more than any single-lever evidence predicts

All 12 far misses come from the two `w4a16-quantized … skip8` cells. Both sit at
**27.8 ms, flat**, across a 33× context range (431 → 14,357 tokens) and across a
window range of 128 → 1024. Nothing else in the run is that insensitive.

The model is not what is wrong here. Both the fit (17.1–19.6 ms) and a plain
linear extrapolation from the measured quantized points agree:

| quantized, R1 | draft_chain |
| --- | --- |
| woff / skip0 (single) | 23.34 ms |
| woff / skip4 (held out) | 20.42 ms |
| w256 / skip4 (held out) | 21.71 ms |
| **w1024 / skip8** | **27.76 ms** ← expected ≈ 18.8 |
| **w128 / skip8** | **27.82 ms** ← expected ≈ 18.8 |

Skipping 8 layers is **6 ms slower** than skipping 4. The slope everywhere else
is ≈ −2.9 ms per 4 skipped layers, in both the target-matching and the quantized
family:

| target-matching, R1 | draft_chain |
| --- | --- |
| woff / skip0 → skip4 → skip8 | 29.98 → 27.21 → 24.21 |
| w128 skip0 → w128 skip4 → w256 skip8 | 31.31 → 28.22 → 25.39 |

So the anomaly needs all three levers at once. `quant × skip4 × window` (21.7 ms)
behaves. `target-matching × skip8 × window` (25.4 ms) behaves. Only the
three-way combination breaks, and it breaks super-additively.

This is a genuine composition effect and is the thing D1 exists to find.

**Not yet explained.** Both boots capture cudagraphs identically
(`FULL_AND_PIECEWISE`, 78 vs 86 PIECEWISE dispatches, 0.54–0.55 GiB), so it is
not a graph-capture fallback. The flatness in context says the chain is bound by
something that does not touch KV. The profiler already instruments
`step_build_attn_md`, `step_input_buffering`, `draft_forward`, `step_sample` and
`wc_replay` inside `draft_chain`, but Round 1 recorded only the `draft_chain`
total, so the decomposition needs a re-run of these two cells with sub-region
capture. That is the next measurement, and it is cheap — 2 boots.

## Against the pre-registered criterion

D1 as registered has two parts: interval coverage, **and** zero false
eliminations under `(K+1)/q_lo < 1 + epsilon_arm`, `epsilon_arm = 0.015`. Only
coverage is computed here; `d1_exercised` is `false` in the record because W14/D
has no scored surface, so this run reports coverage only and grants no Round-2
authority.

The false-elimination count was not computed, so it is not claimed. What the
data does bound is the direction: **0 of 21 misses fall below the band.** A false
elimination requires the model to *over*predict cost, and every miss here
underpredicts it. On this evidence the errors are all in the conservative
direction — the selector would fail to prune, never wrongly prune.

## Also worth recording

`kappa_kv` is **negative at R1** (−1.010e−11) while positive and consistent at
every other regime (8.8e−12 – 3.7e−11). At 431 tokens the KV term is too small
relative to the window and weight terms to be identifiable, and the fit absorbs
it with the wrong sign. The R1 fit should be treated as unidentified in KV even
though it reports `fitted: true`.

## Excluded cell

`target-matching/woff/skip8` was registered as held out but varies a single axis,
so it is not a composed configuration and is also in the fit set — predicting it
would be predicting a point the model was fitted on. It is excluded from D1 and
recorded in `heldout_scope.excluded_invalid` with that reason. The prereg
artifact still carries all eight; it was registered before data and is not
rewritten. The generator flaw was requiring balanced marginals without asserting
that ≥2 axes vary; a test now pins that every scored cell varies ≥2 axes and is
not a fit point.

## Artifacts

- `data/w98_g98b_authorization_v5.json`
- `data/g98_b_v5/{round1_result,d1_fits,d1_predictions}.json`
- `data/g98_b_v5/{singles,heldout}/` — 15 boots, 6 regimes each
- superseded: `g98_b_v4` (measured the engine step), v1–v3 (see the v5 commit)
