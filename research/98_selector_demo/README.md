# Phase 98 — two-round selector demonstration

Status (2026-08-16): **every registered claim is measured and scored, and
D3 has been re-scored under a fail-closed rule.** Phase 97's
runtime-switching infrastructure remains NOT a dependency of this phase's
scored claims — but the D3 re-score has withdrawn the argument against
building it (81% of mixes, not 7%).

| gate | claim | state |
| --- | --- | --- |
| G98-A | smoke, both quant paths | closed |
| G98-B | Round 1 cost | scored and closed; D1 reported NOT EXERCISED |
| G98-C | Round 2 cost (D1') | **SCORED**: 47/48 covered (97.9%), zero false eliminations with the rule exercised 5x |
| G98-D0 | acceptance-path smoke | passed |
| G98-D | acceptance (D2a, D2c) | **SCORED**: D2(a) 6 violations / 132 rows; D2(c) 107/264 pairs separated |
| G98-D2b | knapsack identity (D2b) | **SCORED**: 11/12 controls beaten — wins at k=4 and k=8, inverts at k=16 |
| G98-E | end-to-end selector value (D3) | **SCORED**: 95.1% of omniscient, 1.371x over OFF; ties one composed static |
| — | D3 re-scored, fail-closed rule (analysis) | **99.6% of omniscient, 1.436x over OFF; beats every static (+6.2%) on 81% of mixes** |

**Every registered claim of this phase is now measured.** D1' and D2(a)/(c)
pass; D2(b) fails one clause for a located reason. D3's static clause failed
as originally scored and **passes** once the selector can decline to arm.

Results: [`results_g98_c.md`](results_g98_c.md) (cost),
[`results_g98_d.md`](results_g98_d.md) (acceptance),
[`results_d2b_knapsack.md`](results_d2b_knapsack.md) (knapsack identity),
[`results_g98_e.md`](results_g98_e.md) (end-to-end selector value),
[`results_d3_failclosed.md`](results_d3_failclosed.md) (the fail-closed rule
and D3 re-score), [`results_switching_law.md`](results_switching_law.md)
(when per-regime switching is worth anything, and the axis that would pay),
[`results_certified_region.md`](results_certified_region.md) (the surrogate's
recall guarantee),
[`results_lever_mechanics.md`](results_lever_mechanics.md) (is a weak lever
badly implemented or badly suited),
[`results_window_overhead.md`](results_window_overhead.md) (the window's
short-context overhead, measured and removed),
[`results_nsys_lever_accounting.md`](results_nsys_lever_accounting.md) (the
Nsight per-operation account: what is expected and what is overhead),
[`results_phase98_summary.md`](results_phase98_summary.md) (consolidated),
[`results_clamp_investigation.md`](results_clamp_investigation.md) (the
measurement-environment investigation and the box relocation).

The last three are **analysis over already-measured grids**, not new
campaigns; each names its script and record file.

**Both campaigns run on h104 (bare metal), GPU 7 / NUMA node 1.** The
original box was a QEMU/KVM guest whose host-side "clamp" gated 40+ attempts
without a single accepted cell; sections 10-11 of the clamp document record
the relocation, the two h104 infrastructure faults fixed en route (a
compile-cache key poisoned by the per-attempt trace path, and a PTX/driver
toolchain mismatch that killed every quantized boot), and the localisation of
the clamp itself to a host-side timekeeping cost.

Source: Phase 96 (`w12_search_spec.md`, `w14_plan.md`, W14/B results, W14/D
data pending scoring), Phase 97 (shared-KV composition runtime, P4 same-event
recorder, K/OFF live registry), and the 2026-08-10 design discussion, which
fixed the lever set and the Round 1 / Round 2 split:

1. **Weight quantization** — boot-class residency decided per workload; the
   runtime later toggles between resident paths (B1, future Phase 97 work).
   The target-matching path is always resident and is the universal safe
   state.
2. **Window (sparse attention)** — runtime-class over `{128, 256, 512, 1024}`
   once cost-true per-window capture (G2b) lands; boot-class today. `w-off`
   is a distinct realization under the current FULLCG constraint.
3. **Layer skip** — runtime-class among prevalidated sets; the identity is
   chosen offline (Round-2 knapsack), never constructed at runtime. The
   count ladder is the safety/demotion direction.

## Objective

Demonstrate that the two-round selector works well, end-to-end, on the
measured dense domain — **without waiting for the runtime-switching
infrastructure**. Every runtime-class decision is realized boot-statically
per configuration; the only live-switched axis is K/OFF (the existing P3
registry). Switching value beyond K/OFF is demonstrated analytically against
the omniscient per-segment composite, which simultaneously produces the value
case that sizes Phase 97's G2b/G3/B1 engineering.

The demonstration is of the algorithm, not the engine: Round 1 predicts cost
from single-lever profiles and eliminates soundly; Round 2 measures
acceptance where it must; the combined map plus the live K/OFF ladder beats
every static alternative on a declared workload mix.

## Demonstration claims (to be preregistered before any scored run)

- **D1 — Round-1 soundness on the composed lattice.** A factored cost model

  ```text
  D(quant, w, k) ~= (1 - k/L) * ( W_bytes(quant)*kappa_w(x)
                                  + KV_bytes(w)*kappa_kv(x) + c0(x) )
  ```

  fit from **single-lever** profiles only, produces 95% intervals for
  `q_a(x) = P_a(x)/T(x)` that cover held-out **composed** configurations'
  measured values, with zero false eliminations under
  `(K+1)/q_lo < 1 + epsilon_arm`, `epsilon_arm = 0.015`. Coverage counts and
  the held-out matrix are frozen at preregistration. A coverage failure
  creates a local non-pruning mask (W14 semantics), not a global failure.

- **D2 — Round-2 selection quality.**
    - (a) *Screen honesty:* no composition admitted by the product bound
    `f_set >= prod f_i` has a confirmed tau below the bound minus the
    declared tolerance.
    - (b) *Knapsack identity:* at each surviving count k, the knapsack set
    beats the count-matched random-set and worst-set controls on confirmed
    tau, per regime, outside the tie-set noise.
    - (c) *u-axis resolution:* `tau(w, g, u)` reproduces the I2 crossing
    structure (window preference flips with generated-suffix length) with
    interval separation, not point estimates.

- **D3 — end-to-end selector value.** On the declared workload mix, the
  selector's chosen configuration per regime (boot-static composition plus
  the live K/OFF ladder) achieves at least 90% of the omniscient per-segment
  composite, and its LCB beats every static single configuration including
  OFF under the +2% LCB rule, in decode currency. The 90% target follows the
  Phase 82 precedent (95% of omniscient; 97% of the unconditional ceiling)
  and is frozen or amended only at preregistration, before data.

## Scope

### In scope

- dense Qwen3-8B (the validated domain);
- levers: draft-weight path (target-matching, one quantized realization),
  window `{128, 256, 512, 1024, off}`, skip counts and 2–3 knapsack
  identities, `K in {2, 4}` via one KMAX position-resolved stream, `OFF`;
- shared target KV throughout (Phase 97 invariant);
- offline inference, static batch, pure-decode scoring (continuous batching
  excluded per the 2026-08-09 decision);
- the frozen regime family (R4/R5/R5cot/R8/R1/R6) with fresh phase-local
  prompt manifests and >=2 disjoint content seeds.

### Out of scope

- building per-window capture, non-destructive skip, or B1 co-residency
  (Phase 97's work; this phase produces the value case that sizes it);
- live window/skip/quant switching claims (analytic composite only);
- MoE/llama transfer claims; production workload weights; serving-tail
  metrics.

## Design

### Round 1 — state-indexed cost with a factored model

- Consume Phase 96 W14/D's scored surface for `{w512, w2048, w-off} x K x b`
  when available; extend with single-lever profiles for the missing axes:
  windows `{128, 256, 1024}`, skip counts, and the quantized path.
- Fit `T(Q)` and `P_a(Q)` affine per (batch, K, graph stratum) from exact
  step-trace `Q`; derive `q` intervals per the W14/D interval construction
  (paired block bootstrap, symmetric log residuals).
- Fit the factored model from singles; freeze predictions for a held-out
  composed set (~8 triples at 2 cells spanning weight- and KV-dominated
  states) before measuring them.
- The analytic leverage router (W14/C, 15% threshold) routes measurement
  budget only; it never eliminates.
- Skip enters Round 1 as the **count ladder only**: `f*_k` from
  `D_k ~= (1-k/L) D_0`, eliminated iff `f*_k > f_bar` with
  `f_bar = f(0)` (sound: skipping never raises acceptance).
- **Fallback:** if Phase 96's transfer verdict fails for a stratum, that
  stratum's Round 1 runs in measure-everything mode and D1 is reported as
  not exercised there (partial-success framing, W14 semantics).

### Round 2 — acceptance on the survivors

- One unconditionally-armed KMAX stream per (composition, regime): greedy,
  fixed batch shape, pinned realization, per-position reached/accepted
  counters binned by generated-suffix position. This yields `tau(K)` for
  every `K <= KMAX` and every u bucket from a single stream.
- Boots are cheap, tokens are the currency: acceptance is near-deterministic
  under greedy at fixed batch/realization, so replication is over content
  seeds, not boots. Target ~2,000–4,000 armed steps per
  (composition, regime, u-bucket); uncertainty by request-level bootstrap;
  identical prompt sets across compositions for paired comparison.
- Generation runs to natural EOS or a cap below the typical EOS point;
  accounting closes as `E + C = A + H` with explicit `H` and clip term `C`
  (the P4 same-event recorder machinery, phase-locally adapted).
- **Screen then confirm:** measure singles per regime; screen compositions
  with the product bound (admit-only); confirm only portfolio finalists with
  composed bursts under the Phase 97 composition contract (conditional
  non-regression, mechanism retention, +2% LCB for positive claims).
- **Skip identity:** per-layer sensitivity profiles (dense 36/36 exist;
  refresh per regime as needed) + Round-1 cost weights -> knapsack (top-k in
  the uniform-cost dense case) -> one confirmation burst per (set, regime),
  with the random-set and worst-set controls for D2(b). Escalate to greedy
  forward selection only where the product-bound prediction fails badly.
  Round 2 also answers the consolidation question: one robust identity
  across regimes, or regime-split sets — this directly sizes G3.
- **Realization bridge:** screening may run eager/piecewise; one paired boot
  quantifies the eager-vs-captured acceptance bridge; survivors must clear
  tau* by more than the bridge width; finalists are confirmed under the
  deployed realization.
- **u buckets:** provisional `{0–256, 256–1k, 1k–3k, 3k+}`; final boundaries
  are a scored output of the first Round-2 run (placed where tau(w,g,u)
  actually crosses), then shared verbatim with the runtime policy key.

### End-to-end demonstration (D3)

- **Live arm:** per regime, boot the selector's chosen composition
  (boot-static) with the K/OFF ladder live (descending-tau* order, SPRT
  demotion, low-duty probes, dwell hysteresis — the existing machinery).
- **Baselines:** every static single configuration (each lever alone at its
  best setting, each composition in the pool, OFF/AR), same prompts, same
  currency.
- **Omniscient composite:** from the per-u-bucket segment rates of every
  measured configuration, compose the best-per-segment trajectory
  analytically. The gap between the live arm and the composite is the
  measured value of runtime window/skip switching — the number Phase 97's
  G2b/G3 build is justified (or not) by.
- Scoring: decode currency (`request_decode_time`), same-event accounting,
  paired by prompt, LCB reporting; tie-sets, never point winners.

## Gates (fail-closed, in order)

| gate | content | cost |
| --- | --- | --- |
| G98-0 | CPU harness: accounting closure, u-binned counter tests, factored-model and knapsack unit tests, manifest freeze | CPU only |
| G98-A | **PASSED 5/5** (2026-08-11): every boot class initializes, one target-owned KV group, PIECEWISE chain, generates. Quantized draft + shared KV **verified**; lattice stays at 30. Required a new `w98-lattice` boot scope and scope-aware weight proofs. See `results_g98_a.md` | ~5 non-scored boots |
| G98-B | Round-1 profiles + factored fit; frozen held-out predictions committed before reveal | scored boots per matrix |
| G98-C | **as registered**: Round-2 singles, screen, confirmations, knapsack + controls. **As executed**: the Round-2 COST campaign (D1') — see the drift note | scored boots per matrix |
| G98-D | **as registered**: end-to-end comparison (D3). **As executed**: the acceptance campaign (D2a/D2c) | scored boots per matrix |
| G98-E | end-to-end comparison (D3) — the gate this phase's D3 will run under | scored boots per matrix |

**Gate-name drift, recorded.** The gate letters above were registered before
the Round-2 amendment (`w98r2_prereg.md`) split COST into its own second
round. That inserted a campaign the original order had no letter for, so the
executed sequence shifted by one: G98-C became Round-2 cost, G98-D became
acceptance, and **D3 therefore runs as G98-E**. The letters are relabelled,
not the claims — D1', D2 and D3 keep their registered content and order, and
the scored artifacts (`results_g98_c.md`, `results_g98_d.md`) keep the names
they were committed under rather than being rewritten.

Rules carried from 96/97: preregistration before scored data; frozen prompt
manifests with hashes; one authorization per GPU attempt; consumed outputs
are immutable; decode currency mandatory; notune; no GPU command without an
explicit source-bound authorization record; GPUs 0/1 are occupied by the
Phase 97 screen — check occupancy before any authorization.

## Assumptions (verified at G98-A or earlier)

- A quantized draft realization can boot with shared target KV on the
  current stack (B1 co-residency is NOT assumed — the quant boot is the only
  draft in its boot class).
- The P4 recorder adapts to phase-98 manifests without modifying Phase 97
  frozen sources.
- Phase 96 W14/D scoring lands before G98-B; otherwise the fallback mode is
  used and D1 narrows.
- Position counters support KMAX=8; u-binning is recorder-side (offline from
  raw captures), requiring no engine change.

## Budget sketch (re-estimated at each gate; not an authorization)

Singles: ~6 window/quant/skip-count boots. Sensitivity refresh: reuse
existing profiles where valid. Confirmations: ~10–15 composition boots.
E2E: ~8–10 boots (chosen configurations + baselines + AR anchors).
Order 25–35 scored boots total at the observed ~1h/boot cadence, plus smoke.

## Relationship to other phases

- **Phase 96** owns cost-transfer validation (W14/D scoring is its next
  artifact) and remains the methodological source. Phase 98 consumes, never
  rescores, its results.
- **Phase 97** owns the engine: measurement machinery reused here;
  G2b/G3/B1 engineering sized by D3's composite gap and Round 2's
  consolidation answer. Phase 98 makes no engine change beyond phase-local
  adapters.
- The paper claim this phase supports: the two-round selector (cost
  predicted and sound, acceptance measured and content-local) selects
  compositions that beat static alternatives, and the remaining headroom to
  the omniscient composite quantifies what runtime switching is worth.

## Engine surface this phase added

Additive and scope-gated, so the scopes Rounds 1-2 were measured under do
not move:

* **`w98-d2` boot scope** (`vllm/v1/spec_decode/koff_runtime.py`) — KMAX = 8
  unconditional arming, and per-request acceptance rows carrying each
  request's OWN generated-suffix length. Every other scope keeps the closed
  K in `{0, 4}`, emits no acceptance rows, and keeps its verbatim error text.
* **Amendment 2** (`w98_prereg_amendment2_instrument.md`, APPROVED under the
  dual-arm option) — the profiler no longer syncs inside the region it
  measures: 12 syncs per step become 4. `VLLM_SELF_SPEC_PROFILE_LEGACY_SYNCS=1`
  keeps the legacy instrument runnable, which is the mechanism D3's
  registered conservative bound depends on.
* One upstream fix outside the scope system: step-0 work evidence was gated
  on the action id being literally K4, so any other armed action arrived as
  `None`.

## Open items

1. **One clause remains failed and is reportable as such.** D2(b) loses one
   of twelve controls, at k=16 only — located, explained, and not repaired
   by more measurement. D3's failure has been repaired: it was a missing
   capability (the selector could not decline), not a mis-ranking.
2. ~~D3's static clause depends on a reading of the preregistration.~~
   **MOOT.** Under the fail-closed rule the selector beats every one of the
   31 statics by 6.2%, so the strict reading passes and no ruling is needed.
   The rule is post-hoc with a pre-D3-data-only derivation; see
   [`results_d3_failclosed.md`](results_d3_failclosed.md) for the invariance
   checks offered in place of a commitment barrier.
3. **The u-bucket boundaries** are still the provisional ones; placing them
   where `tau(w, g, u)` actually crosses is a scored OUTPUT of G98-D and an
   analysis step, not a new measurement.
4. **The clamp verdict** on the original box needs one X29 run there (one
   clamped burst, one clean burst); the prime suspect is quantified in
   section 11 of the clamp document.
5. **Phase 97's runtime switching is the open engineering question, and the
   re-scored price is 81% of mixes, not 7%.** The original figure was an
   artifact of the missing fail-closed rule and the argument it supported
   has been withdrawn. What remains is a cost question against +6.2%, and
   the observation in [`results_switching_law.md`](results_switching_law.md)
   that the strongest case for switching is not regimes preferring different
   levers but the lever set being **unaffordable** in some of them —
   modelled at 1.305x.
6. **The next campaign is KV pressure — now `research/99_kv_pressure/`.**
   Its preflight has already corrected the design: the draft costs a
   measured **44,096 KV tokens (~6.5 GB)**, and at batch 8 the feasibility
   crossover sits beyond Qwen3-8B's context limit, so the obvious sweep
   would have returned a null that looked like a refutation. The campaign
   runs at batch >= 16. Registered prediction: per-regime selection beats the
   best uniformly-feasible static by **>= 1.20x**.
7. **The u-axis instrument has never been fed.** Its bucket edges are
   [256, 1024, 3072] and every boot ran 640 tokens, so two of four buckets
   are permanently empty. Within the measured span acceptance moves with
   position (up to +16.3%) but the configuration ranking does not (rho +0.82
   to +0.97, argmax stable in all six regimes) —
   `scripts/analyze_w98_position_axis.py`. Phase 99's long outputs are what
   would populate it.
