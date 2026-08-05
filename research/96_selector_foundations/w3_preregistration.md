# W3 — pre-registration: the numbers that will judge the selector build

Committed BEFORE W4/W5/W6 produce any data. This document fixes the
currency, the measurement protocol, and the go/no-go thresholds so that
the two-round pipeline (W6) and hidden switching (W7b) are built only if
pre-declared numbers clear pre-declared bars. Screening estimates from W2
appear as context and are NOT confirmatory evidence.

## 1. Currency (binding, from results_w2.md)

- **Model validation and all thresholds below: S_dec** — per-request
  decode-rate ratio (TPOT currency), engine-core timers
  (`request_decode_time_seconds`). It is what `S = (1+fK)/(KR+1)`
  predicts.
- **Deployment claims: S_e2e**, always reported next to S_dec with the
  bridge terms (φ, queue, straggler) measured, never inferred.
- Engine configuration for every scored run:
  `enable_flashinfer_autotune=False` (W1), no pinning, ITERS ≥ 4.

## 2. Measurement protocol (binding; supersedes results_w1.md §protocol)

1. **Episode rejection, batch ≥ 8 cells**: reference = max round across
   the configuration's boots; reject rounds >5% below; ≥2 surviving
   rounds or the boot re-runs.
2. **b1 cells** (new — W2 found max-anchoring too aggressive there):
   reference = **second-highest** round across the cell's boots; reject
   rounds >5% below it; ≥2 surviving rounds. (Robust to one lucky-fast
   round; tolerant of b1's legitimate spread.)
3. **Cross-boot certification** (new — a long episode can cover a whole
   cell, results_w2.md F4): every scored cell needs either (a) a
   replicate boot whose surviving-round median agrees within 2%, or
   (b) an in-boot anchor (the AR arm of the same boot) that matches its
   own cross-boot reference within 2%. A cell with neither is marked
   UNCERTIFIED and re-run before use.
4. Rejected/uncertified counts are reported in every artifact.

**Noise floor (measured, W1/W1b under this protocol): 1.0%** — the max
observed certified replicate spread (spec 0.7%, AR 0.4–0.9%). All margins
below are measured against this floor.

## 3. Estimator corrections (binding for W6+ runtime scoring)

- The live accept estimator must not re-inject the optimistic 1.0 default
  per new request (F5's +0.1 standing bias). For W6's scoring, f comes
  from measured accept counters over the scored window (no EMA at all).
  The RUNTIME estimator redesign (optimism only at cold start, pooling
  default = current EMA) is W8 scope.
- R constants come only from notune-audited cells (W4's re-audit). No
  lottery-era table feeds any W6 decision.

## 4. The W6 go/no-go gate (binding)

Definitions, per architecture, over the fixed regime suite
{R4, R5, R5cot, R8, R1, R6}:

- **Baseline B**: the best SINGLE static composition (one window, skip,
  quant, KMAX booted once) combined with a per-regime oracle OFF gate:
  score(regime) = max(S_dec(static, regime), 1.0). The static composition
  is chosen by exhaustive search over the measured candidates — the
  baseline is given every advantage.
- **Treatment T**: per-regime composition selection from the Round-1
  shortlist (Round 2's output), same OFF gate.

**Build W6's runtime (and W7b) only if, on notune-certified data:**

- `mean_regimes(T − B) ≥ +2.0%` in S_dec (2× noise floor), **or**
- any single regime shows `T − B ≥ +5.0%` while `mean(T − B) ≥ 0`.

Additionally, per-STEP switching (W7b as opposed to per-regime W7a)
requires W4's measured transition cost amortized over the observed
switch cadence to be `< 0.5%` of throughput — otherwise switching stays
per-regime even if the selection gate passes.

**Fail outcome (pre-declared)**: if neither branch passes, the two-round
pipeline is NOT built; the deliverable is the static-plus-gate
deployment, whose value is already banked (MLA −47% avoided at b1;
llama R1/R6/R8 losses avoided; the gate needs only the OFF decision to
be right).

## 5. Screening context (NOT confirmatory)

From W2's certified cells, the WINDOW-ONLY per-regime envelope over
best-static+gate is small: dense **+0.3%**, llama **+0.9%** — below the
gate. The thesis therefore rides on the FULL composition pool (skip ×
quant × window × K from the re-audited map): **W5 must PREDICT
`mean(T − B) ≥ 2%` from audited cells before W6 spends GPU time.** If
the audited map cannot predict a passing margin, the fail outcome is
taken without running W6. (Precedent for a real margin: phase 80's
KnapSpec triple-composition +10% at its cell — composition, not window,
is where C2 found large wins.)

## 6. W4 measurement plan under this pre-registration

1. **Transition cost**: 95/`run_e1p_probe.sh` two-point duty
   discriminator (dutyA 2/64 vs dutyB 8/256 — same duty, 4× flip count):
   per-flip cost = slope over flips; per-armed-step cost = intercept.
   Prediction: per-cycle cost ~20 ms (phase-95 estimate) — measured value
   replaces the scheduler's hand-set 2% hysteresis.
2. **Parked cost**: OFF-scored engine vs true AR at b8–b32 (the 82-E0
   gap): parked ≠ 1.0 becomes a per-cell constant in the model.
3. **llama C2 re-audit (notune)**: re-measure R at a stratified sample of
   llama cells (b1/b8/b32 × short/mid/long ctx, both windows, K2/K4).
   Prediction (F6): audited R ≈ table R / 1.46 uniformly; if the
   inflation is NOT uniform, the whole llama table regenerates, not just
   a scale factor.
