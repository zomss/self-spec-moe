# Phase 98 — two-round selector demonstration

Status: **design registered; no preregistration frozen, no GPU command
authorized. Phase 97's runtime-switching infrastructure (per-window graphs,
B1 co-residency, non-destructive skip) is under construction and is NOT a
dependency of this phase's scored claims.**

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
| G98-A | smoke: every required boot class initializes and passes shared-KV/alias/sampler preflights — **including the quantized-draft + shared-KV assumption, which is unverified on the current stack** | ~5 non-scored boots |
| G98-B | Round-1 profiles + factored fit; frozen held-out predictions committed before reveal | scored boots per matrix |
| G98-C | Round-2 singles, screen, confirmations, knapsack + controls | scored boots per matrix |
| G98-D | end-to-end comparison (D3) | scored boots per matrix |

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

## Expected next artifact

`w98_prereg.md` + `data/prereg/` — the frozen D1/D2/D3 matrices, prompt
manifests with hashes, held-out splits, tolerances, and the G98-0 CPU
harness with its tests. No GPU work precedes it.
