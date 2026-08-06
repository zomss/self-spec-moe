# W10 — MoE Stage-B re-measurement under equal work (pre-registration)

Written BEFORE any W10 run. Purpose: settle the C1 MoE row, whose
"8/33 beats AR" count rests on a protocol W9 showed to be unsound for
marginal cost-side cells.

## Why the existing row cannot stand

1. **4 of its 8 wins are drain-contaminated** (W9b screen): R6 b32
   (S = 1.288), R1 b64 (1.278), R2 b32 (1.172), R8 b8 (1.025) — all
   with the arms' output-length distributions diverged >10%.
2. **An independent careful re-measurement contradicts the largest
   one**: W7 measured MoE R6 b32 at S = 0.857 (OFF) against Stage B's
   1.288, and W9 retracted the R2 b32 arm (1.087 → 0.928/0.943 under
   equal work).
3. **The artifact cuts BOTH ways**, so checking only the wins is not
   enough: five LOSING cells are contaminated and within 15% of 1.0
   (R6 b64 0.985, R8 b64 0.932, R5 b8 0.915, R5cot b8 0.888, R2 b8
   0.882) and could flip UP. The honest count needs the whole surface.
4. The Stage-B MoE artifacts also predate `notune` (`tune: null`), so
   they additionally carry W1's 40% autotune boot lottery.

## Protocol

Full Stage-B MoE surface re-run: 5 arms × 33 cells (9 regimes ×
batches {1,8,32,64}, minus not-enough-prompt cells), TP2 on GPUs 0+1,
sequential. Arms replicate `run_stage_b.sh` exactly:

| arm | draft | K | window | extra |
|---|---|---|---|---|
| off | off | 0 | — | — |
| w4a16_k2 | W4A16-INT4-sym | 2 | — | SHARED, no Machete |
| w4a16_k3 | W4A16-INT4-sym | 3 | — | SHARED, no Machete |
| win2048_k3 | self | 3 | 2048 | winplain |
| win8192_k3 | self | 3 | 8192 | winplain |

Two protocol changes, both fixes: `G93_TUNE=0` (notune) and
`G93_FIXED_LEN` per regime. ITERS=4 (episode rejection needs ≥4;
Stage B used 3).

**Per-regime fixed lengths** = the Stage-B off arm's natural p50,
median across batches, so each regime KEEPS its character (R7 stays a
23-token regime, R5cot stays a 1030-token regime) while both arms do
identical work:

    R1:280 R2:205 R3:265 R4:815 R5:330 R5cot:1030 R6:76 R7:23 R8:1250

Disclosed: fixed length is itself an idealization — real serving has
continuous arrivals, where the batch neither drains to zero nor stays
perfectly uniform. What it guarantees is EQUAL WORK between arms,
which is the minimum condition for a meaningful ratio and exactly what
natural EOS violates (the arms generate different text; W9).

Scoring: per cell, best S over the four spec arms vs the off anchor;
2nd-highest-round anchor with 5% episode rejection.

## Pre-registered predictions

- **P-W10a**: the MoE win count DROPS from 8/33. Point prediction
  **2–5 of 33**. Basis: three of the four contaminated wins are
  expected to fall (W7 already refutes R6 b32 and W9 refutes R2 b32),
  partially offset by flips from the five near-1 contaminated losses.
- **P-W10b**: R6 b32 lands **below 1.0**, reproducing W7's 0.857
  rather than Stage B's 1.288. This is the sharpest single test — the
  two protocols disagree by 50% on one cell.
- **P-W10c**: at least one of the five near-1 contaminated LOSSES
  flips above 1.0. If none does, the artifact is one-directional in
  practice and the W9b sweep's two-sided caution was over-stated.
- **P-W10d**: the surviving wins concentrate at b32/b64 on
  short-output regimes, preserving C1's qualitative "OFF-dominant,
  quant wins at high batch" character even as the count falls. If
  instead the wins vanish entirely, the MoE column becomes a pure
  gate-only architecture and c3's MoE verdict (gate everywhere)
  and c1's row agree exactly.

Outcome is publishable either way: it either repairs a count or
converts the MoE column into a clean second instance of the
acceptance-independent OFF story already proven for MLA below b32.
