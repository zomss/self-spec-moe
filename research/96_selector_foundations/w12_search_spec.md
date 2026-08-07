# W12 — the search process, specified

Consolidates W0–W11 into one precise statement of the selector. Every
constant cited is measured and traceable; every open input is marked
OPEN. Supersedes the prose in `w6_design.md` §1–3 (which remains as the
design rationale and its amendment history).

## 0. Objects

Cell `c = (b, ctx)`; regime `g` (content character); lever
`ℓ = (quant, window, skip-set, kv-quant, realization)`; depth `K`.
A **configuration** is `(ℓ, K)`; the action space also contains `OFF`.

Throughput identity, in the generalized form (W8):

    S(ℓ,K,c,g) = τ(ℓ,K,g) / ( K·D(ℓ,c)/T(c) + V(K,c) + C(c)/T(c) )

    τ = 1 + Σ_{i<K} p_i     (expected accepted tokens per armed step)

- `T(c)` AR step time. `D(ℓ,c)` draft step time. `V(K,c)` verify cost in
  target-step units (=1 for dense at low batch; coverage-scaled for
  sparse; → K+1 as the target becomes compute-bound). `C(c)`
  per-armed-step serving-stack overhead.
- Everything in the denominator is **cost**; only `τ` carries content.

## 1. Why the staging exists (the dichotomy)

| quantity | varies with | offline-identifiable | measured evidence |
|---|---|---|---|
| D, V, C, T | cell, arch, realization | **yes** | R reproduces calibration→deployment at 1.00× (W4b) |
| τ (via p_i) | regime × **content** | **no** | ρ ∈ [0.41,1.23] across content draws; llama 35% separability failure |

Consequence, measured: freezing the offline argmax costs **+1.75%**
(dense) / **+7.05%** (llama) mean, **+37.1%** worst — the error floor.
So cost decides elimination, acceptance decides selection, and nothing
arms without measured acceptance.

## 2. Round 1 — cost-only elimination (sound; no acceptance anywhere)

**Inputs (all offline, f-free):** profiled `T(c)`, `C(c)`, `V(K,c)`,
and `D(ℓ,c)` for each lever in the pool, per realization.

### 2.1 Stage 0 — cell kill

The maximum draft budget at cell `c` and depth `K`, using only
`τ ≤ K+1`:

    D*(K,c) = ( K+1 − V(K,c) − C(c)/T(c) ) / K

Let `D_min(c)` be the cheapest **realizable** draft in the pool. Then

    cell c is OFF for every configuration   iff   max_K D*(K,c) < D_min(c)

Special case `V + C/T ≥ K+1` ⇒ `D*<0`: verify alone exceeds the maximum
possible token yield, so no draft whatsoever can pay.

*Guarantee.* Sound: `S` is monotone in `f` and the test uses the
`f = 1` upper bound, so a killed cell cannot contain a winner.
Asymmetric by construction: **Stage 0 can only prove OFF, never ARM.**

*Validated* (W8 terms, K=4): eliminates MLA b1 (D\*=0.615 vs D/T 1.69)
and MoE b1 (0.658 vs 1.64); admits MLA b32 (0.848 vs 0.78) and MoE b32
(0.793 vs 0.74, tight — matching W10's +1.3–3.6% marginal wins).

### 2.2 Stage 1 — (lever, K) pair elimination

For surviving cells, eliminate the pair `(ℓ,K)` iff
`D(ℓ,c)/T(c) ≥ D*(K,c)`.

K-aware by construction: `V` depends on `b·(K+1)`, so different `K` sit
at different points of the verify curve and are eliminated
independently. **D\* tightens as `b` grows** (V grows), which is the
formal statement of "the draft lever must get cheaper at larger batch".

### 2.3 Stage 2 — the skip COUNT ladder (never the identity)

Count → cost is physics (each skipped layer removes ≈1/L of draft
compute); identity → acceptance is content-dependent. So Round 1 emits
the surviving **counts** only. Count `k` has draft cost `D_k`; its
break-even acceptance is `f*_k` solving `S=1`. Eliminate count `k` iff
`f*_k > f̄`, where `f̄` is a sound upper bound on achievable acceptance:
`f̄ = 1` (always sound) or, tighter, the measured no-skip `f(0)` —
sound because skipping never raises acceptance (`f(k) ≤ f(0)`).

### 2.4 Realization

`realization ∈ {piecewise, scratchpad}` enters as a **cost** axis with a
measured decode-length interaction (W4c): equivalent within 2% at
512-token decodes, but 0.846 vs 1.420 at ~72-token decodes. Round 1
therefore runs **per realization**, and the pool is keyed by it.

**Output of Round 1:** per cell, a candidate pool of `(ℓ,K)` pairs plus
the surviving skip counts — with the guarantee that the true optimum is
still in it.

## 3. Round 2 — acceptance measurement on the survivors

**Measured per (regime, composition), NOT per cell** — acceptance is
cell-independent (CV 2.2–3.3%) while cost is not (CV 14.5%). One
profile therefore spans the whole batch × ctx grid.

**Protocol:** unconditional bursts (no policy mixing), per-position
accept counters, natural EOS, ≥2 disjoint content seeds. Position
counters give `τ(K) = 1 + Σ_{i<K} p_i` for **every** K from one armed
stream — so one boot calibrates the entire depth ladder.

**Skip identity — KnapSpec with measured profits.** At each surviving
count, choose *which* layers by a knapsack whose weight side (per-layer
cost) is Round-1 physics and whose **value side is measured here**
(per-layer sensitivity bursts), composed under the C2 sound bounds
(`f_set ≥ ∏ f_i` admits), then one confirmation burst of the composed
set. This is the precise contrast with KnapSpec: same combinatorial
structure, but the profit vector is measured per regime rather than
scored offline — which the dichotomy and the 35% separability failure
say is mandatory.

**Output:** per regime, a ranked ladder over the surviving pool, with
tie-sets = the ε-optimal set at the spent budget. `OFF` is always a
member.

*Guarantee.* `f̂` concentrates as a Bernoulli rate; the identity maps
f-intervals to S-intervals with Lipschitz constant `K/(KR+1) ≤ 1.2`, so
an ε-optimal lever per regime with probability `1−δ` costs
`O(log(|pool|/δ)/ε²)` drafted tokens. Total sample complexity
`O(#levers + #confirmations)`, independent of the number of
compositions.

## 4. Serving — ranked ladder with sequential demotion

**The regime is never identified explicitly.** At serving time the
observable key is `(b, input length, generated length)` — the cell, not
the content. So:

1. the **cell** selects which Round-2 ladder to use (a prior over
   configurations), and
2. **live acceptance** discriminates within it.

The ladder therefore performs online model selection, with Round 2 as
its prior. This is what dissolves the I2 problem (regimes sharing a
cell but wanting opposite levers) without adding a map axis.

**Policy.** Walk the ranked ladder; arm the first option whose
break-even test passes on *measured* `τ` (same identity as Round 1, so
a mis-ranked option cannot be armed below break-even); demote when the
live accept CI implies `S<1` (an SPRT on fixed telemetry — per-position
counters, no optimistic re-injection, F5); re-promote via low-duty
probes. Keyed on generated length as well as batch/input, because both
cost and acceptance live on the suffix axis (I2, W4c).

*Guarantee.* Switching is free (per-flip ≈ 0, W4a) and mis-arming costs
≈1.06 step-equivalents per armed step, so
`regret ≤ probe-duty × duty-cost + detection-delay × drift-loss`, every
constant measured. Probe rate is a solvable optimization, not a tuned
constant.

*Measured value:* the ladder recovers **+25%** over the biased argmax
it replaces (190.5 vs ~152 tok/s, 97% of the unconditional ceiling), and
holds K=4 on 1858/1858 steps at b1 with zero flapping.

## 5. What the search is optimizing (goal restated, user 2026-08-07)

Not "win everywhere versus AR" — at compute-bound cells verify alone
costs (K+1)× and **no** draft can pay, so OFF is the correct answer, not
a failure. The objectives are:

1. **Coverage** — the number of cells that can be profitably armed, and
2. **Gate correctness** — arming exactly those (the error floor
   measures the loss when this is done offline-only), and
3. **Dominance over alternative self-spec methods at every regime** —
   which holds by construction if their configuration is in our pool,
   since Round 2 measures rather than assumes.

Reporting "N/33 vs AR" alone misframes correct OFF decisions as
failures.

## 6. Measurement cost

| stage | cost | frequency |
|---|---|---|
| Round 1 profiling | O(#levers) step-time profiles per arch × realization | once per (arch, hardware) |
| Round 2 acceptance | O(#survivors + #confirmations) bursts per regime | once per (arch, regime set) |
| Serving | per-position counters, already in the step | continuous, free |

## 7. OPEN inputs

- **`V` and `C` are measured only on MLA/MoE** (W8). The dense and
  llama columns need the same profiling (~1 GPU-h) before Stage 0/1
  can be applied there; today those columns' Round 1 still runs on the
  banked-R form.
- **The profiler perturbs armed-step cost by 25–29%** (W8d), so
  Round-1 inputs derived from it are conservative-biased in the wrong
  direction for OFF proofs. An unperturbed profiler (CUDA events/nsys)
  is required before Stage 0 is certified rather than demonstrated.
- **Per-layer sensitivity profiles** exist for dense (36/36 layers);
  other arches would need theirs before the skip-identity knapsack runs.
- **P-W8c untested**: no out-of-sample validation of the constructive
  denominator (predict K=2 or b64 from terms measured elsewhere).
