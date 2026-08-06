# W11 — llama + q3_32b Stage-B surfaces under equal work (pre-registration)

Written BEFORE any W11 run. Completes what W10 started: the two
remaining c1 rows whose counts inherit the artifacts W9/W10 measured.

## Why these two rows need it

W10 found **two** independent inflation mechanisms in Stage B, and only
one of them is architecture-specific:

1. *The drain* (batch > 1, arms generate different-length text). W9b
   flags 13/24 multi-request llama cells and 11/24 q3_32b cells with
   >10% tail spread between arms.
2. *A lottery-slowed AR anchor* (ALL cells, W1's autotune variance).
   Stage-B MoE AR rates ran 3–14% slow; since S = spec/AR, this
   inflates every S uniformly. R6 b1 was the clean control (+9.7% with
   identical output length, drain-immune). **This mechanism is
   architecture-general** — nothing about it is MoE-specific — so both
   remaining rows are exposed even where the drain screen is clean.

Neither row is at qualitative risk (both surfaces are rich and
multi-winner), but their exact "N/33" counts are not currently
defensible.

## Protocol

Identical to W10. Full surfaces, canonical Stage-B arms only (the
llama `.PREFIX` variants are excluded — they are not in
`run_stage_b.sh`'s ARMLIST and are not part of c1's count):

| arch | TP | arms | GPUs |
|---|---|---|---|
| llama | 1 | off, w4a16_k2, w4a16_k4, w8int8_k2, win512_k2, win2048_k4 | 0 and 1, two parallel lanes |
| q3_32b | 2 | off, w4gptq_k4, w4gptq_k6, win512_k4, skipb2_k4 (skip 7,16) | 0+1, sequential |

`G93_TUNE=0`, ITERS=4, per-regime `G93_FIXED_LEN` = the arch's own
Stage-B off-arm natural p50 (median across batches):

    llama   R1:229 R2:327 R3:218 R4:1032 R5:56 R5cot:655 R6:227 R7:25 R8:737
    q3_32b  R1:326 R2:294 R3:282 R4:983 R5:446 R5cot:1204 R6:102 R7:23 R8:1231

## Pre-registered predictions

Calibration: applying a uniform anchor lift to the existing margins
gives llama 22/20/17 and q3_32b 24/19/14 survivors at 3/9/14%. The
llama surface is more robust (max margin 1.683, median 1.183) than
q3_32b's (max 1.248, median 1.148), which is compressed near 1.

- **P-W11a (llama)**: count falls from 24/33 to **17–22**. Predicted
  point: 20.
- **P-W11b (q3_32b)**: count falls from 25/33 to **14–22**, a WIDER
  band because its margins are compressed — a median win of 1.148
  cannot absorb a 14% anchor lift. Predicted point: 19.
- **P-W11c**: q3_32b loses proportionally MORE than llama, because its
  win margins are tighter (median 1.148 vs 1.183 and a far lower
  maximum). This is a comparative prediction that does not depend on
  the absolute deflation.
- **P-W11d (the qualitative claim, the one that matters)**: both rows
  keep ≥3 distinct winning configs and no single config wins the whole
  surface — i.e. C1's "no universal lever" claim is untouched by the
  correction. If either surface collapses to a single winner, C1's
  central claim weakens on that architecture and must be restated.
- **P-W11e (two-sidedness)**: at least one LOSING cell flips up on each
  arch, as it did for MoE (2 flips). If none flips on either, the
  artifact is one-directional in practice on rich surfaces.

Disclosed: as in W10, the Stage-B → W11 delta conflates three protocol
changes (drain fix, notune, fixed-length/ignore_eos). W10 measured
ignore_eos to be acceptance-neutral (mean τ shift −0.2%), so the delta
is cost-side; separating drain from anchor requires the b1 cells, which
are drain-immune by construction and therefore isolate the anchor term
on each arch exactly as R6 b1 did for MoE.
