# W12 — the search process, specified

Consolidates W0–W11 into one precise statement of the selector. Every
constant cited is measured and traceable; every open input is marked
OPEN. Supersedes the prose in `w6_design.md` §1–3 (which remains as the
design rationale and its amendment history).

## 0. Objects

Cell `c`; regime `g` (content character); lever
`ℓ = (quant, window, skip-set, kv-quant, realization)`; depth `K`.
A **configuration** is `(ℓ, K)`; the action space also contains `OFF`.

**Cell definition (user, 2026-08-07) — continuous, and a TRAJECTORY.**
A request is not a point in the grid; it sweeps one. A 16k-input /
2k-output workload traverses cells 16k → 18k as it generates, and the
selector must be indexed so Round 1, Round 2 and serving all read the
same axis. Two axes are required, because they are not the same
quantity:

| quantity | index | why |
|---|---|---|
| cost (D, V, C, T) | **total context** (per-step: #scheduled requests + total scheduled KV tokens) | weight reads scale with request count, attention with KV length |
| acceptance (τ, p_i) | **regime × GENERATED length** | I2: widening the window buys +0.30 accept at gen 3072 but +0.02–0.09 at gen 512 — value tracks the self-generated suffix, not total ctx |

Collapsing both onto total context would merge a 16k-input/2k-gen
request with a 2k-input/16k-gen one: identical cost, different
acceptance. That is exactly the I2 failure, reintroduced.

The serving key `(b, input length, generated length)` determines both
axes, which is why it is the key.

**Granularity is free, because cost is interpolated not tabulated.**
`T(ctx) ≈ a + b·ctx` (constant weight reads + linear attention); a
full-KV draft has the same shape; **a windowed draft is FLAT in ctx**
(it attends over the last W tokens regardless). So 3–4 profiled context
points per lever fit two parameters and `D*(K,c)` evaluates at any
granularity. Corollary worth exploiting: since windowed `D` is flat
while `T` grows, `D/T` DECREASES with context, so the crossover context
at which the window lever begins to pay is SOLVABLE in closed form
rather than tabulated — the mechanism behind C1's observed "window wins
at long context".

**Trajectory ⇒ switching, and only runtime-class levers can follow it.**
If the optimal lever changes at 17k, following the trajectory means
switching mid-request. Window and K are runtime-switchable; quant,
skip-identity and realization are boot-class (W0). So boot-class levers
are chosen once against the DISTRIBUTION of trajectories (the portfolio
problem, §4) and runtime-class levers follow the trajectory inside that
choice.

*Measured scope*: our grid is the homogeneous special case
(total KV = b × ctx). Continuous batching mixes contexts within a step;
the (requests, total-KV) form is the general index and degenerates to
`(b, ctx)` for what we measured.

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

### 2.0 The Round-1 output is a REQUIRED-ACCEPTANCE label (user, 2026-08-07)

Round 1 should not emit a filtered list. The denominator is pure cost,
so every configuration carries a number Round 1 can compute exactly:

    τ*(ℓ,K,c) = K·D(ℓ,c)/T(c) + V(K,c) + C(c)/T(c)

τ* is **the acceptance that configuration must achieve at that cell to
break even**, and then

    S = τ_measured / τ*

so Round 2's job is to supply τ, and selection is a division. Nothing
is thresholded in Round 1 except the impossible: τ* > K+1 cannot be
satisfied (τ ≤ K+1), which recovers the Stage-0 cell kill and the
(lever,K) elimination below as COROLLARIES rather than separate rules.

*Validated out-of-sample* (results_w8.md §3b): predicted τ* = 4.71
(MLA b32) and 4.80 (MoE b32) against unprofiled actuals 4.64 and 4.81
— +1.5% / −0.3% — and the ARM/OFF call follows directly (MLA τ=4.94 >
τ* → ARM, actual S=1.064; MoE τ=4.47 < τ* → OFF, actual S=0.928).

**Pool construction: SPAN τ*, do not maximize predicted S.** Round 1
cannot rank by S (it has no acceptance), but it can guarantee the pool
holds an option at every acceptance level — a cheap draft with τ*≈1.5
alongside an aggressive one with τ*≈4.7. Picking the top-N by any
predicted score clusters them; a spanning set means that whatever
acceptance turns out to be, a matched configuration exists. This is
the principled answer to "how do you choose a pool without knowing
acceptance".

**Ladder consequence.** The serving fallback chain should be ordered by
**descending τ*** — when live acceptance falls, demote to a
configuration that needs less of it. That ordering is regime-robust,
where ordering by one regime's measured S is not.

**Trajectory in τ*-space.** As context grows, a windowed draft's cost
stays flat while the target's grows, so **τ* FALLS for windowed
configurations** — an option too expensive at 2k becomes affordable at
16k. But acceptance moves too, and not always favourably: a fixed
window covers a shrinking fraction of the relevant suffix, so τ can
fall as well (I2: widening buys +0.30 accept at gen 3072 vs +0.02–0.09
at gen 512). S = τ/τ* is therefore a ratio of two quantities that both
move along the trajectory, in regime-dependent proportion — which is
precisely why the window must be measured per regime and tracked live,
not solved once.

### 2.05 Pool construction is CELL-DEPENDENT and leverage-aware (W13)

A lever's maximum cost leverage is the share of the step's memory
traffic attributable to the term it modifies:

    quant        -> weight bytes         leverage = weight share
    window, kvq  -> KV bytes             leverage = KV share
    layer skip   -> whole layers (BOTH)  leverage = k/L, every cell

Both shares are computable ANALYTICALLY per cell from the model config
and batch x ctx — no measurement needed. Round 1 should therefore
include only levers whose leverage at that cell clears a threshold, and
pass sub-threshold levers straight to Round 2 as acceptance-only. This
makes Round 1 both cheaper (never profile a lever that cannot matter
there) and sharper (no ranking on differences smaller than the error).

Measured (Qwen3-8B): weight/KV share is 96.6/3.4 at b1-1.1k ctx,
77.9/22.1 at b8-1.1k, 21.2/78.8 at b8-14.5k. D/T spread across three
window settings tracks it exactly: 2-7% at the low-KV cells (noise),
97-110% at b8/14.5k. W6's winners independently obey the rule — its R1
winner uses a window (w2048) that cannot even bind at 1.1k ctx, so it
is effectively skip-only, while R5/R5cot win with an active w512.

### 2.1 Stage 0 — cell kill (a corollary of §2.0)

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

- **BLOCKER (W13, 2026-08-07): the sync-bracketed profiler is not
  accurate enough at short steps.** Measured bias in τ* is +11…+17%
  mean and, fatally, DIFFERENTIAL between compositions at one cell (20
  points at R1 b8), which breaks the ranking Round 1 must produce:
  2/6 top-1, 1/6 exact order on dense. Bias tracks step length (worst
  at b1 ≈5 ms, near zero at b8/14k), the signature of the profiler's
  fixed per-region CUDA syncs — which is why W8 §3b passed at b32
  (11.5 ms steps) and this fails. A CUDA-event profiler is the fix.
  Until then, τ* is usable only on long-step (high total-KV) cells —
  which is also exactly where W13 found cost discriminates
  compositions at all (τ* spread 30–49% at b8/14k vs 1–8% at b1).
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
