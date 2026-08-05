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

**Skip-count rule (user, 2026-08-06)**: within the skip lever, Round 1
decides only the COUNT ladder, never the identity. Count → R is pure
physics (each skipped layer removes ~1/L of draft compute); identity →
f is content-dependent and Round-2-only. Since more skipping always
lowers R, cost-only reasoning cannot argmax the count — Round 1 instead
emits the SURVIVING counts by break-even: count k survives iff its
break-even f = R_k is plausibly below the measured no-skip acceptance
(sound, because skipping never raises f: f(k) ≤ f(0)). The current pool
froze count at {0, 2} (the phase-93 search's b2), where the R saving is
3–6% and sometimes inside noise; the ladder {0, 2, 4, 8} opens the axis.

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

**Skip-identity stage (KnapSpec with measured profits)**: at each
surviving count, Round 2 selects WHICH layers via the knapsack whose
weight side (per-layer cost, uniform) is Round-1 physics and whose
VALUE side (per-layer acceptance cost) is measured here — per-layer
sensitivity bursts, composed under the C2 sound bounds
(`f_set ≥ ∏ f_i` admits), then ONE confirmation burst of the composed
set. KnapSpec profiled these values offline on calibration content; the
dichotomy (and the measured ρ spread) says the profit vector must be
per-regime measurement instead. Measurement cost: identity is
boot-fixed (W0), so either a one-time per-arch L-boot sensitivity
profile (valid iff per-layer RANKING transfers across regimes — test on
a 3-layer subset first) or the G3 gated-passthrough measurement mode
(all layers in one boot; engine change).

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

### Measured (scripts/error_floor.py → data/w6/error_floor.json)

| arch | Floor A mean | worst cell | where the loss lives |
|---|---|---|---|
| dense | **+1.75%** | +3.6% (R5) | R4 +2.9% (arms where live says OFF — as predicted), R5/R5cot +3.3–3.6% (window mis-picks) |
| llama | **+7.05%** | **+37.1% (R1)** | R1: offline prefers w512 (pred 1.168) whose deployed realization is 0.745 vs w2048's 1.116; plus R5cot +4.0% |

Prediction scored: dense exactly in the predicted band with the
predicted mechanism; llama EXCEEDS the band, and its worst cell is a
window mis-pick at a short-ctx b1 cell (the miscalibrated-grid region),
not an arm/OFF error — the floor has two mechanisms, not one. Floor B
coincides with A where scorable; 3/12 unrestricted picks are s-none
variants never measured live (UNSCORED, coverage disclosed).

Caveat, per the rule: live values are deployed-runtime realizations
(policy path), so llama R1's 0.745 folds the runtime's mis-arming into
the offline pick's cost — which is the honest accounting: an
offline-only selector ships exactly that runtime.

**The dichotomy's impossibility half now has its number: a purely
offline search pays +1.75% (dense) / +7.05% (llama) mean against a
measurement-informed selector restricted to the SAME three actions —
and up to 37% on a single regime.**

## 5. Work items

| # | item | status |
|---|---|---|
| 1 | error floor (§4) | this commit + `scripts/error_floor.py` |
| 2 | dense Round 2 (real-acceptance calibration at R1/R5/R5cot pools, ~2 boot-classes) | next GPU round — decides the paper's positive number |
| 3 | formal section: soundness lemma, sample complexity, regret decomposition, portfolio formulation | writing, no new math needed |
| 4 | telemetry: per-K accept counters; remove optimistic pooling re-injection | engine change, feeds 2 and serving |
| 5 | serving ladder implementation (SPRT demotion, probe-rate optimization) | after 2 |
