# Design discussion record — 2026-08-21

The researcher's direction, verbatim in substance, and the assessment that
opened this phase. Recorded so later work can be checked against what was
actually decided rather than a drifted memory of it.

## The direction (researcher)

1. Target the RL rollout scenario mainly; the Phase 98 refined-grid eval is a
   pilot study / evidence collection for it.
   1. RL rollout is offline inference with really long output generation, so
      this workload needs lever switching to maximize throughput.
   2. At the beginning of a generation, relatively short context and large
      batch favor layer skip over the window; as context grows, the window
      becomes preferable.
2. High-level design: Round 1 is cost modeling with calibration; Round 2 is
   acceptance modeling; at runtime the system adapts at inter- and
   intra-rollout steps.

## Assessment (agent, agreed)

* 1.1 is supported by the phase-98 data: the largest measured mix gaps (up to
  +19.8% over the best static) are exactly batch-drain pairs, and every
  rollout traverses that axis.
* 1.2 is supported and computable. Two measured facts compose into it: a
  window cannot bind until context exceeds w+16 (s35), and the window's value
  grows with context (3.15 ms at 4K to 21.2 ms at 12.7K, s37). From the s42
  coefficients the crossover is closed-form:

      u*(B) ~= window + 16 + f_win / (kappa_kv * KV_bytes * B)
             ~= window + 18,800 / B     [tokens, this model/box]

  B=32 crosses at ~1.6K context, B=8 at ~3.4K, B=1 not until ~19K. The
  schedule is genuinely dynamic in (u, B): the drain pushes the crossover out
  while context growth runs past it.
* 2 matches what survived phase 98: the s38 calibration discipline, measured
  u-resolved acceptance with natural-length truncation, and the confirmation
  budget (top-3 -> 99.6% of omniscient). The analytic shortcut forms are
  refuted as rankers (s48, s49) and must not reappear in Round 1.
* One argument the direction gains for free: self-spec's draft derives from
  the policy's own weights, so it tracks RL training automatically.

## Gaps and risks

As enumerated in README.md (gaps 1-7, risks 1-4). The two that gate
everything: temperature (all acceptance is T=0; RL samples at T~1) and the
dwell-time law (assumed switching value has failed twice at +1.4% and +2.37%;
the trajectory integral must be priced analytically before engineering).

## Sequence agreed

E1 trajectory simulation (free) -> E2 T=1 acceptance -> E3 runtime demo on
K+skip only -> E4 runtime window (G2b) only if E1 justifies it.
