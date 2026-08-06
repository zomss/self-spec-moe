# W10 results — the MoE row re-measured under equal work

Pre-registration: `w10_moe_remeasure.md` (commit a52672648, before any
run). Scorer `scripts/score_w10.py`, committed before the data landed.
Artifacts `data/w10/` (5 arms × 33 cells, complete, notune,
per-regime fixed length, ITERS=4).

## All four predictions confirmed

| # | prediction | outcome |
|---|---|---|
| P-W10a | win count drops from 8/33 to 2–5 | **CONFIRMED: 5/33** (at the top of the band) |
| P-W10b | R6 b32 lands below 1.0 | **CONFIRMED: 0.954** (Stage B claimed 1.288) |
| P-W10c | ≥1 near-1 contaminated LOSS flips up | **CONFIRMED: 2** (R1 b32, R7 b32) |
| P-W10d | surviving wins stay at b≥32 | **CONFIRMED: 5/5** |

## The corrected MoE surface

Five cells beat AR, and every one is **marginal**:

| cell | S | arm | Stage B said |
|---|---|---|---|
| R5cot b32 | **1.036** | w4a16_k3 | 1.180 |
| R1 b32 | 1.019 | w4a16_k3 | 0.980 (was a LOSS) |
| R1 b64 | 1.018 | w4a16_k3 | 1.278 |
| R7 b32 | 1.013 | w4a16_k3 | 0.944 (was a LOSS) |
| R2 b64 | 1.003 | w4a16_k2 | 1.034 |

Five Stage-B wins fell (R2 b32, R3 b32, R3 b64, R6 b32, R8 b8) and two
losses rose — the artifact is **two-sided**, as W9b cautioned and
P-W10c pre-registered.

**Deployment reading.** Against the policy tables' arming rent of 1.5%,
only **3 of 33 cells** justify arming (R5cot b32, R1 b32, R1 b64), and
the largest margin in the whole surface is +3.6%. MoE remains an
OFF-dominant architecture whose wins do not pay for the switching
machinery — the qualitative claim in c1 survives, the count does not.

## Two independent inflation mechanisms in Stage B

The Stage-B → W10 deltas are NOT purely the drain fix. Isolating them:

1. **The drain** (batch > 1 only, W9's mechanism). Confirmed by the
   flagged cells: R6 b32 −25.9%, R8 b8 −24.2%, R2 b32 −16.6%.
2. **A slow AR anchor** (all cells, including drain-immune b1). The
   notune AR rates are 3–14% FASTER than Stage B's:

   | regime | Stage-B AR b1 | W10 AR b1 | Δ | fixed len vs natural |
   |---|---|---|---|---|
   | R6 | 219.1 | 240.3 | **+9.7%** | 76 → 76 (**identical**) |
   | R2 | 218.2 | 249.5 | +14.3% | 188 → 205 |
   | R4 | 203.1 | 228.3 | +12.4% | 789 → 815 |

   **R6 b1 is the clean control**: same regime, same output length,
   drain-immune — and the AR anchor still gains 9.7%. That is the W1
   autotune lottery, and since S = spec/AR, a slow anchor inflates
   EVERY S in the arm uniformly.

So Stage-B MoE's 8/33 was inflated twice over: by the drain at
batch > 1, and by a lottery-slowed AR denominator everywhere.

## ignore_eos does not move acceptance

Mean τ shift across all 33 cells: **−0.2%** (max |shift| 2.9%).
Generating past EOS produces different text but not measurably
different predictability, so the fixed-length protocol is
acceptance-neutral and the S changes above are purely cost-side. This
retires the main risk of the equal-work protocol.

## Consequences

- **c1's MoE row**: "8/33" → **5/33**, winning configs W4A16-K2/K3 only
  (win8192-K3 no longer wins any cell), character unchanged
  ("OFF-dominant; quant wins concentrate at high batch").
- **c3's MoE verdict**: W7/W9's "arms nowhere" was measured on
  R1/R2/R6 at b1/b8/b32 under natural EOS. W10's equal-work sweep finds
  5 marginal wins, 3 clearing arming rent, all at b≥32. The precise
  statement is: **no MoE cell wins by more than 3.6%, and none of the
  cells W7 examined arms** — the gate is still the right answer, but
  "no cell beats AR anywhere" would be too strong.
- **The notune correction is architecture-general.** It affects every
  Stage-B AR anchor, not just MoE's. Dense (32/33, large margins) and
  MLA (clean by construction, corroborated by W7) are not at risk of a
  qualitative change, but their exact counts inherit the same
  uncertainty. Re-measuring llama and q3_32b under this protocol is the
  open item; neither carries a headline that a few-percent anchor shift
  can move.
