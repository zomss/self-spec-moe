# W6 — the selector, redesigned (user proposal 2026-08-06) with its formal skeleton

Supersedes the W5-era W6 scope. W3's gates remain binding. This document
is committed BEFORE the error floor (§4) is computed.

## 0. The spine: an identifiability dichotomy

`S = (1+fK)/(KR+1)` factors throughput into:

- **R — offline-identifiable.** Transfers calibration→deployment at
  1.00× (W4b), roofline-fittable (v2r, 3 params/arch),
  content-independent. Predictable physics.
- **f — online-only.** Regime-dependent (ρ 0.41–1.23), and on llama
  composition-entangled (35% separability failure): NOT identifiable
  from calibration content. Cheap to measure live (per-token accept
  events, thousands/s).

Design principle: predict what transfers, measure what doesn't, allocate
each quantity to the cheapest stage that identifies it. The three stages
below are that allocation; each carries a classical guarantee whose
constants we have measured.

## 1. Round 1 — sound cost-only elimination (no acceptance anywhere)

Eliminate (comp, K) at cell (batch, len) iff
`sup_f S = (1+K)/(K·R_lo+1) < 1 + ε`, with `R_lo` the FAVORABLE end of
the measured R band (replication spread + noise + N-bias 0.83–1.00) and
per-realization (piecewise default; scratchpad separately — the
short-decode cliff, W4c).

**Guarantee (soundness lemma)**: S is monotone in f, so the bound holds
over ALL acceptance behaviors — Round 1 never eliminates the true
optimum, up to the stated R-band. Completeness is empirical (it removes
all of MLA/MoE below b32, high-batch K4, expensive quants). Output: the
candidate pool per (input-length, batch) cell. Grid amendment: add
sub-2000-ctx cells (R1/R6/R8 miscalibration, W5).

## 2. Round 2 — real-acceptance calibration on the surviving pool

Measure f per (regime, composition) — NOT per cell (f is
cell-independent, C2) — with: per-regime content matched to deployment,
UNCONDITIONAL bursts (no policy mixing), per-K accept counters, natural
EOS included, ≥2 content seeds (G6). Compose S from measured-f ×
modeled-R for ranking; finish with ONE end-to-end confirmation of each
regime's picked winner (catches interaction terms — the realization
cliff was invisible to the composition).

**Guarantee (sample complexity)**: f̂ concentrates as a Bernoulli-rate
estimate; the identity maps f-intervals to S-intervals with Lipschitz
constant K/(KR+1) ≤ 1.2, so an ε-optimal lever per regime with
probability 1−δ needs O(log(|pool|/δ)/ε²) drafted tokens. The tie-set is
the formal ε-optimal set at the chosen budget. Boot cost is compressed
by the W0 action space (runtime-switchable window/K share boots).

## 3. Serving — ranked ladder with sequential-test demotion

Key = CURRENT state (batch, input length, **generated length** — I2/W4c:
both accept and cost live on the suffix axis), updated as generation
proceeds. Per key: the Round-2 ranked ladder (OFF always a member).
Policy: run the top lever; DEMOTE to the next when the live accept CI
implies S < 1 (an SPRT, using fixed telemetry: per-K counters, no
optimistic re-injection — F5); RE-PROMOTE via low-duty probes.

**Guarantee (regret decomposition)**: switching is free (W4a: per-flip
≈ 0) and mis-arming costs ~1.06 step-equivalents per armed step
(measured), so regret ≤ probe-duty × duty-cost + detection-delay ×
drift-loss — every constant measured (drift: phase 91/92). The probe
rate is a solvable optimization, not a tuned constant.

**Portfolio constraint (the KnapSpec generalization)**: the boot-class
choice + resident-graph pool is a budgeted portfolio selection —
maximize expected throughput over the regime mixture subject to the
capture/residency budget (S2), with online recourse inside the pool.
Two-stage stochastic knapsack; Rounds 1–2 solve stage one, serving
solves stage two.

## 4. The offline-only error floor (pre-registered rule; item 1)

The dichotomy's impossibility half, quantified from committed data: what
a PURELY offline search (C4 acceptance, no live measurement) must lose.

Rule, per (arch ∈ {dense, llama}, regime ∈ W2's six):

- **Offline pick** = argmax over actions of
  `S_pred = (1 + f_C4·K)/(K·R+1)` at the regime's mapped cell (R and
  f_C4 from the oracle pairs, inverted as compile_from_c2; ρ = 1 — no
  live correction, that is the point). OFF (S=1) is an action.
- **Floor A (action-restricted, always scorable)**: actions = {OFF,
  deployed-comp w512, deployed-comp w2048} with offline K choice; score
  the pick by its LIVE S_dec (W2 episode-rejected medians; llama R5
  w2048 uses the W4c-certified piecewise value; OFF = 1).
  `loss = S_live(live best action) − S_live(offline pick)`.
- **Floor B (unrestricted)**: offline argmax over the full surviving
  pool; live-scorable only when the pick is a measured action, else
  reported as UNSCORED with its identity (an honest coverage statement).
- Report per-cell losses, mean and worst per arch. Every input is a
  committed artifact; the live table's provenance is cited inline in the
  script.

Prediction (pre-registered): the floor is dominated by
arm-where-live-says-OFF errors (dense R4-class cells, ρ ≈ 0.41), with
mean loss 1–3% and worst-cell loss ≥ 5% on at least one arch.

## 5. Work items

| # | item | status |
|---|---|---|
| 1 | error floor (§4) | this commit + `scripts/error_floor.py` |
| 2 | dense Round 2 (real-acceptance calibration at R1/R5/R5cot pools, ~2 boot-classes) | next GPU round — decides the paper's positive number |
| 3 | formal section: soundness lemma, sample complexity, regret decomposition, portfolio formulation | writing, no new math needed |
| 4 | telemetry: per-K accept counters; remove optimistic pooling re-injection | engine change, feeds 2 and serving |
| 5 | serving ladder implementation (SPRT demotion, probe-rate optimization) | after 2 |
