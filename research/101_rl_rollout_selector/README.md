# Phase 101 — the lever selector for RL rollout

Source phases: 98 (`results_refined_lo.md` sections 29-49,
`results_refined_summary.md`), 100 (protocol and Campaign 1), 97 (runtime
switching engine, deferred), 75 (EfficientRollout reproduction, the prior-art
baseline).

Opened 2026-08-21 after the 2026-08-21 design discussion, which redirected the
work: **the target scenario is RL rollout**, and Phase 98's refined-grid arc is
reframed as the pilot study / evidence collection for it.

## Objective

Maximize rollout throughput for RL training via runtime lever switching in
self-speculative decoding: select and ADAPT the draft configuration
(quantization, KV window, layer skip, draft depth K) across a rollout, rather
than fixing one configuration per job.

## Why RL rollout is the target

* Offline inference: throughput-only currency, no latency SLO.
* Very long outputs (CoT traces, 5-25K tokens): the LO cell -- exactly the
  rollout shape -- is where Phase 98 measured the largest wins (1.66-1.75x
  stock) and where the winning arm moves with batch.
* The batch DRAINS within every rollout and the context GROWS within every
  generation, so a single job traverses the (batch, context) plane that Phase
  98 showed wants different arms at different points: the largest measured
  mix gaps (up to +19.8% over the best static) are precisely b1-vs-b32 pairs.
* Self-spec's draft derives from the policy's own weights, so it tracks
  training automatically -- no draft model to retrain between iterations.

## Objectives (registered 2026-08-22, researcher)

1. **Beat the single static baseline with a big margin — over 10%.**
2. **Test on real RL rollout**, demanding much longer context than the
   phase-98 grid measured.
3. **Test on other models.**

Objective 1 raises the bar above the earlier E1 criterion (under ~3% fails
the thesis): the target is now +10%, and the static-point analysis says the
equal-mix ceiling at the measured operating points is +7.28% — so reaching
10% requires the trajectory (intra-rollout switching), the longer contexts of
objective 2 (the window crossover grows with context), or the K axis. E1 is
the first pricing of that path.

## Design (working)

* **Round 1 -- cost modeling with calibration.** The section-38 discipline:
  fit at the longest context and largest batch the deployment reaches, two
  batches minimum, per weight version. Analytic simplifications are known to
  misrank (sections 48-49); the full model is the ranker.
* **Round 2 -- acceptance modeling.** Measured per content at KMAX, truncated
  at the natural output distribution, u-resolved. One depth-8 profile yields
  tau(K) for all K.
* **Runtime adaptation.**
    * *Intra-rollout*: switch levers as context grows and the batch drains.
    Working hypothesis (supported by the fitted model): skip/quant early --
    short context, weight traffic dominates, windows cannot bind -- window
    later, with the crossover at approximately
    `u*(B) ~= window + 18,800/B` tokens.
    * *Inter-rollout*: cheap re-calibration as the policy trains (acceptance
    drift is unmeasured).

## What Phase 98 already established (evidence base)

| finding | where |
| --- | --- |
| best arms vs stock at the rollout-shaped cell: 1.66-1.75x | s45 |
| no static is best: 7 winners over 14 points; ceiling over best static +7.28% equal-mix, up to +19.8% on drain-shaped mixes | s45, mix analysis |
| two-round search works with a confirmation budget: top-3 confirms to 99.6% of omniscient | s48, s50 |
| cost model calibration discipline (long context, two batches, per family) | s37, s38, s42 |
| acceptance protocol (per content, natural-length truncation, tau(K) free from depth-8) | s35 |
| K=4 is usually wrong; K=2 wins 3 of 4 measured arms; K=8 always loses | s49 |
| engine tax located: host-GPU serialization; parked fix worth 2.5%; async scheduling +19% for off at LO but catastrophic at SS b32 | s46, s47 |
| analytic speedup forms misrank even with perfect acceptance inputs | s48, s49 |

## Known gaps to the target scenario

1. **Temperature.** RL samples at T~1; every Phase 98 acceptance number is
   greedy T=0. If T=1 reorders the lattice, sections 35+ are calibration for
   the wrong operating point. Highest-priority measurement.
2. **No trajectory evaluation.** All measurements are static (cell, batch)
   points; the value of an omniscient switching SCHEDULE over a draining
   rollout has not been computed even analytically.
3. **Runtime-class coverage.** K is live-switchable and skip is
   runtime-class; **window and quant are boot-class** (G2b/B1, deferred when
   switching was priced at +1.4%). The intra-rollout hypothesis needs the
   window to come ON mid-run.
4. Switch overhead itself: unmeasured on this branch.
5. RL-scale batch (100s-1000s) and the quantized draft's 6.1 GB residency
   against KV headroom: unmeasured past b64.
6. Acceptance drift across training iterations: unmeasured.
7. Prior-art baseline: the RL claim must compare against EfficientRollout
   (phase 75), not only stock vLLM.

## Risks, ranked

1. T=1 reorders the lattice (measure first, one campaign).
2. Dwell-time law: the trajectory integral may concentrate in one arm --
   priced analytically before any engineering (this failed twice before at
   +1.4% and +2.37%).
3. Switch overhead eats the crossover gains.
4. Currency: rollout claims must be whole-job wall clock, prefill included.

## Plan of record

1. **E1 -- trajectory simulation** from the Phase 98 grid: omniscient
   schedule vs best static over an RL-shaped drain (LO prompts, b32 to b1,
   context growth), plus the u*(B) schedule. Free; prices the direction.
2. **E2 -- T=1 acceptance campaign** on the lattice singles (~20 boots).
3. **E3 -- intra-rollout demo on runtime-class levers only** (K + skip,
   window static): measures switch overhead with no new engineering.
4. **E4 -- G2b (runtime window)** only if E1 says the window crossover
   carries the value.

## Decision criteria

* E1 gain over best static < ~3%: the switching thesis fails on this
  workload family; fall back to "selector finds the static" (Phase 98's
  result) and close.
* E2 reorders the lattice: re-run Phase 98's acceptance protocol at T=1
  before anything else consumes it.
* E3 switch overhead > the E1 per-switch gain: coarsen the schedule (fewer
  segments) and re-price.

## Expected next artifact

`results_e1_trajectory.md` -- the analytic schedule value, from existing data.
