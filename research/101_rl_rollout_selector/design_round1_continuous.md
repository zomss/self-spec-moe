# Round 1 for rollout: the continuous trajectory model

Date: 2026-08-22. Design, not measurement. Source: phase 98's
`w98_cost_u.integrated_with_drain` (section 7, calibrated in sections 37-42)
and the acceptance protocol of section 35.

## 1. What already exists, stated honestly

Phase 98's model is **already an integral over the generation**: it splits the
step into batch-shared and per-request parts, walks `u` in segments, applies
`B(u) = #{i : G_i > u}` from the length distribution, and reads `tau` from
u-buckets. The per-(cell, batch) predictions that reached 99.6%-of-omniscient
shortlisting were evaluations of that integral **at a fixed arm**. So "a
better analytic model for rollout" is not a new model -- it is three upgrades
to this one, plus a change of decision variable.

## 2. The formulation

A rollout batch of `N` requests, prompts `p_i`, generation lengths `G_i`
(from the PREVIOUS RL iteration -- the workload repeats, so the length
distribution is data rather than an assumption). Batched decode advances all
active requests in lockstep, so position `u` is shared and the active batch
is the empirical survival function:

    B(u) = N * S(u),   S(u) = P(G > u)

A **schedule** sigma maps the trajectory point to a configuration:

    sigma : u  ->  (quant, window, skip, K)

Advancing `du` costs `du / tau_sigma(u)` engine steps; each step costs

    t_step(a, u, B) = [V_s + D_s(a, u)] + B * [V_p + D_p(a, u)]

with the verify split and the per-request KV column exactly as calibrated
(sections 19, 42). Total decode wall and tokens:

    T(sigma) = INT_0^Gmax  t_step(sigma(u), u, B(u)) / tau_sigma(u)(u)  du
    tokens   = SUM_i G_i                     (workload input, arm-independent)

**The objective is the integral of a pointwise ratio.** Without switch costs
the optimal schedule is the pointwise argmin:

    sigma*(u) = argmin_a  t_step(a, u, B(u)) / tau_a(u)

and because every `D(a, u)` is affine in `kv_positions(w, p+u)` with a single
kink where the window binds, **arm crossovers along the trajectory are
closed-form**. With a per-switch penalty `varsigma`, the problem becomes 1-D
segmentation, solved exactly by dynamic programming over a discretized `u`.

Two structural notes the formulation makes visible:

* **The fixed per-step host cost is divided by tau.** The engine tax is
  per-step, so higher acceptance amortizes it -- the model captures the tax
  interaction without a separate term.
* **The drain bends the crossover.** The window's saving scales with `B`, so
  `u*(B) ~= w + 18,800/B` recedes as the batch drains while `u` runs past
  it. The schedule is decided along the CURVE `(u, B(u))`, not on the plane.

## 3. The three upgrades

**(a) tau continuous in u, without a distributional assumption.** The
measured object is the per-position prefix profile `pos_accepted[i | bucket]`
at depth 8. Interpolate each position's rate monotonically across bucket
centers (PCHIP), truncate at the natural-length distribution (section 35's
rule), and read `tau(K, u)` by prefix sum. Explicitly NOT the geometric
`tau = (1 - alpha^{K+1})/(1 - alpha)`: the geometric shortcut is exactly the
kind of simplification sections 48-49 refuted, and the prefix profile is
measured.

**(b) B(u) from the empirical survival function** of the previous rollout
iteration. This replaces phase 98's per-cell length lists and is the one
place the RL setting is *easier* than the eval setting: section 21 had to
argue the parked arm's lengths were an obtainable proxy; a training loop has
last iteration's actual lengths.

**(c) The schedule as decision variable**, with switch cost. K enters the
schedule but its cost is NOT modeled analytically -- section 49 measured the
`K*D_fwd + V` shortcut mispicking 3 of 4 optima. K's step cost comes from
the measured K-sweep points (K in {2,4}; K=8 excluded, it loses everywhere),
extended by calibration boots if the schedule wants other depths.

## 4. Why intra-rollout switching is legal at all

The draft KV window is a **read-time compaction of the shared, fully-stored
target KV** (`_apply_draft_kv_window` builds a view; nothing is evicted).
Turning the window on or off mid-run therefore requires **no KV migration**
-- the state is always complete. The window is boot-class today only because
of the capture/graph constraint (FULLCG realizations), which is engineering
(G2b), not data-model surgery. Skip changes the layers the draft runs, K
changes the chain length: both stateless for the same reason. **Every lever
in the schedule is stateless with respect to the KV cache**, which is what
makes the pointwise-argmin formulation valid -- no history term in the
integrand.

## 5. What must not be re-simplified

| temptation | verdict | evidence |
| --- | --- | --- |
| `tau*V/(K*D+V)` per-arm scalar | misranks with PERFECT acceptance | s48 |
| `K * D_fwd` for depth cost | mispicks 3 of 4 optima | s49 |
| product bound as ranker | error correlates with arm quality | s48 |
| geometric alpha for tau(K) | untested shortcut of a measured profile | s48-49 pattern |
| single-batch cost fit | kappa_kv silently absorbs the batch | s42 |
| short-context calibration | unidentified coefficients are unbounded | s36-38 |
| ignoring natural-length truncation | tau = 5.000 filler inflation | s35 |

## 6. Validation plan (E1, free)

* **E1a -- fixed-arm reproduction.** The integral at a fixed arm must
  reproduce the measured LO b8/b16/b32 static throughputs from the phase-98
  grid. This is the regression test the s38 model already passes at ranking
  level; the continuous version must not do worse.
* **E1b -- the schedule value.** `T(sigma*)` against `min_a T(a)` (best
  static) over an RL-shaped drain (LO prompts, B0 = 32, empirical lengths),
  with switch cost swept from 0 to pessimistic. The registered decision
  criterion applies: **gain under ~3% at plausible switch cost fails the
  switching thesis** on this family.
* **E1c -- schedule shape.** Report sigma*(u) itself and check it against
  the 1.2 hypothesis (skip early, window later, K falling with the drain).

## 7. Known limits

* Built from T=0 acceptance; E2 (T=1) gates any conclusion.
* tau's batch non-invariance (s22: under 4%, non-monotone) is treated as a
  noise band, not modeled.
* Valid inside the calibration envelope only; rollout stays inside it iff
  stage 0 calibrates at B0 and max context.
* The lockstep-u approximation treats all active requests as sharing one
  position; real batches mix positions (84% of armed steps span a range,
  G98-D). The drain integral inherits this approximation from phase 98,
  where it predicted at 5-13% level error with correct ranking.
