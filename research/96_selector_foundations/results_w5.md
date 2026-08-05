# W5 — Round 1 results: tie-sets, separability, and the W6 verdict

Procedure: `w5_round1.md` (pre-registered), implementation
`scripts/round1_shortlist.py`, outputs `data/w5/`. Scope dense + llama.

## 1. Separability (f-transfer) — PARTIAL

ρ(regime) = f_live/f_C4 at the deployed comp, window-consistency spread:

| regime | dense ρ (spread) | llama ρ (spread) |
|---|---|---|
| R4 | 0.41 (1.6%) | 0.41 (3.0%) |
| R5 | 1.04 (7.1%) | 0.74 (**13.6%**) |
| R5cot | 1.23 (3.2%) | 0.86 (**35.0%**) |
| R8 | 0.56 (7.8%) | 0.42 (**26.1%**) |
| R1 | 1.15 (5.9%) | 1.18 (4.7%) |
| R6 | 1.10 (7.6%) | 0.65 (0.5%) |

Dense: holds within ~5–8%. Llama: FAILS at R5cot/R8/R5 — a real
content × composition interaction in acceptance (window choice changes
HOW MUCH the regime's content differs from C4). ρ itself is the
quantified content gap: real summarization (R4) accepts **0.41×** C4 on
both arches — the largest single correction Round 1 applies.

Deviation disclosed: where separability fails, the implementation widens
the f band by the measured spread instead of the pre-registered
sound-bounds fallback — more permissive in the safe direction (keeps
candidates, weakens elimination only).

## 2. Tie-sets

Sizes per regime — dense: R4 **1 (OFF)**, R5 28, R5cot 19, R8 **1
(OFF)**, R1 21, R6 26; llama: R4 1, R5 30, R5cot 37, R8 1, R1 20, R6
**1 (OFF)**. Large sets are the honest reading of I3: at current
interval widths the map cannot rank inside them.

**Boot-class compression (the W0 payoff)**: tie-set members differ
mostly in (window, skip, K) at fixed quant — runtime-class or cheap-boot
axes. Dense's four non-trivial tie-sets collapse to ~2 boot-classes
(q-hum × skip∈{b2, none}, boot w2048/K4, mask down); llama's to ~2.
Round 2's measurement cost is a handful of boots, not 20–37.

## 3. Calibration against live data (W2's S_dec)

- **Long-ctx regimes: covered.** Dense R5 live 1.27–1.31 ∈ [1.22, 1.65];
  R5cot live 1.65–1.68 ∈ [1.40, 1.62]/[1.37, 1.64] (edge). Point
  estimates overshoot live by ~10–14% (the known N-bias + f̂ bias
  directions), intervals absorb it.
- **Short-ctx regimes: MISCALIBRATED.** R1 live 1.10–1.11 vs interval
  [1.24, 1.56]; R6 live 1.03 vs [1.03, 1.35] (edge-out). The map's grid
  has no cells below ctx 2000; R1/R6 live at ~300–1100 ctx where the
  nearest-cell R (c2000) understates target-step speed → S over-
  predicted beyond the declared bands.
- **Mixed-K f̂ defect**: R8's tie-set is {OFF} while live measured
  S_dec 1.05 — the policy-mixed accept (as-if-K4) underestimates f
  where accept < 3.5. Live accept telemetry must be per-K for Round 1
  to see such cells.

## 4. The W6 gate verdict (w3_preregistration.md thresholds)

| arch | best static B | mean(T−B) | best single regime | letter verdict |
|---|---|---|---|---|
| dense | q-hum_w-512_s-b2 / K2, B_mean 1.274 | +1.9% | **+5.2% (R1)** | PASS (single-regime branch) |
| llama | q-w4a16_w-2048_s-b2 / K2, B_mean 1.111 | +0.3% | +1.7% (R1) | **FAIL** |

**Robustness read: the pessimistic deltas (T_lo − B_hi) are negative at
EVERY dense regime (−20% to −45%).** And dense's deciding regime (R1) is
precisely one of the miscalibrated short-ctx cells (§3).

**Binding conclusion (per the pre-registered clause):**

- **Dense**: the letter-PASS is UNVERIFIED — the pessimistic read fails,
  so W6's first act is NOT pipeline construction but **interval
  shrinkage by targeted measurement** at the deciding cells (R1, R5,
  R5cot): live S_dec for the tie-set's top candidates, ~2 boot-classes,
  under the W3 protocol. That measurement IS Round 2's per-regime
  confirmation, arriving exactly as the user's original design
  specified — the map screens, measurement decides.
- **Llama**: FAIL is taken, and the live evidence independently supports
  it (live w512-vs-w2048 margins at R5cot are ~2%, inside noise+
  transition; R5's big win belongs to the STATIC w2048 choice, not to
  switching). **Llama's deliverable is the fail outcome: static
  q-w4a16_w-2048_s-b2 (piecewise realization, K≤4 runtime) + the OFF
  gate.**

## 5. Round-1 defects found (feed Round 2 and the paper)

1. The compile grid needs sub-2000-ctx cells (or short-ctx regimes are
   Round-2-only). R1/R6/R8 all live there.
2. Live accept telemetry must be per-K (mixed-K EMA corrupts f̂ exactly
   where the gate decision is close).
3. ρ from policy-mixed runs is biased; Round 2's probes should be brief
   UNCONDITIONAL bursts per candidate (clean f, clean R, one number
   each).
4. Separability failure on llama means llama f̂ can never come from C4
   scaling alone — per-regime measurement is mandatory there (which the
   fail-outcome deployment does implicitly: it measures nothing, it
   gates).
