# Campaign 1 interim record — stopped at 31/35 boots by operator request

Record of 2026-08-17. Runner `scripts/run_w100_campaign1.py` under barrier
v2 `efac8380220c2373`; scorer `scripts/analyze_w100_campaign1.py`
(`data/campaign1/campaign1_scored.json`). **Stopped cleanly on operator
request** ("I will tell you when server is ready") with the four lever
arms' b1 boots outstanding: `w4a16/w1024/knap8/magicdec512 __b1`. To
resume: relaunch the runner on a quiet lane box — it skips the 31
completed boot records and runs exactly the missing four.

## Boxes

b8/b16/b32 groups ran on **h103**; b64 and the partial b1 group on
**h104** (box switched after a co-tenant serving deployment occupied
h103's NUMA-1 CPU mask; h104 had cleared). Every score is a within-group
ratio against that group's own stock arm, so no ratio crosses a box.
Cross-group *trend* readings (the off-overhead-vs-batch curve) carry the
known cross-box caveat (task #19).

## Scores (vs stock; LO context-corrected per amendment 1)

| arm | b1 LO | b8 LO | b16 LO | b32 LO |
| --- | --- | --- | --- | --- |
| off | 0.892 | 0.850 | 0.965 | 1.014 |
| base | 0.897 | 0.880 | 0.945 | 1.055 |
| w4a16 | — | **1.099** | 1.014 | 1.090 |
| w1024 | — | 0.911 | 1.271 | 1.484 |
| knap8 | — | 0.972 | 1.029 | 1.154 |
| magicdec512 | — | 0.843 | 1.190 | **1.504** |

| arm | b8 LI | b16 LI | b8 LIO | b16 LIO |
| --- | --- | --- | --- | --- |
| off | 0.933 | 0.984 | 0.925 | 1.001 |
| base | 0.749 | 0.716 | 0.784 | 0.777 |
| w4a16 | 0.708 | 0.615 | 0.764 | 0.680 |
| w1024 | 0.719 | 0.748 | 0.833 | 0.936 |
| knap8 | 0.778 | 0.753 | 0.769 | 0.734 |
| magicdec512 | 0.683 | 0.754 | 0.799 | 0.916 |

(b1 for off/base: 0.86–0.93 across all four cells. SS columns are
withheld: SS runs first in every boot and absorbs arm-dependent
first-touch cost — Marlin autotune, window path — so SS needs a
dedicated re-run with a real warmup before any SS number is quoted.)

## Findings so far

1. **LO x batch is the window family's regime, with a budget crossover.**
   w1024: 0.91 -> 1.27 -> 1.48 across b8/16/32; magicdec512: 0.84 ->
   1.19 -> **1.50**. The tighter 512 budget loses at b16 and wins at b32
   — the budget/batch crossover sits between them, a direct selector
   input and the measured content of B3'a's "wins when batched" clause.
2. **LI (GovReport summarization) is lost by every draft arm at every
   batch measured** (0.62–0.78) while pure engine overhead (off) sits at
   parity — the phase-98 window content-gate (R4 retention 0.535)
   reproduced on real GovReport, now also binding for MagicDec's own
   configuration. Long context alone does not rescue a windowed draft on
   summarization.
3. **Quant does not scale with batch.** w4a16 LO is flat 1.01–1.10
   across b8–b32 while window arms triple their margin; consistent with
   the R6 nsys finding that the Marlin gap widens with batch.
4. **The off path costs 7–15% at b1–b8 and amortizes to parity by b32**
   (LO: 0.89 -> 0.85 -> 0.965 -> 1.014). At low batch the stack's
   not-speculating overhead is the first-order blocker for every arm;
   engineering it away is worth more than any single lever at b1/b8.
5. **The T=0 divergence mechanism is confirmed as batch-composition
   numerics** (amendment 1): at b1, `off` diverged on **1 of 64**
   T=0 requests (vs 57/112 at b8), and `base` at b1 diverged 0/16 on SS,
   rising with length (5/16 LI, 8/16 LIO, 15/16 LO) exactly as K=4
   verify-step numerics compounding over length predicts.
6. **Measurement gates: clean** for all three arms that completed their
   four-point SS sweep (stock, off, base), including across the box
   switch.

## Open items for the resume + analysis pass

* Four b1 boots (the quant sweet-spot claim rests on w4a16 b1).
* SS re-run with a real warmup (order-position artifact).
* Gate refinement to register in the next barrier revision: the 3a sign
  test fires trivially when nonzero deltas are few (off b1 LI: 1 flip =
  100% one-sided). Require a minimum nonzero count (e.g. >= 6).
* base's isolated one-sided flags (SS b8, LO b16) largely dissolved
  under the b1 evidence (identical SS outputs at b1); keep the sign-test
  aggregate in the final analysis rather than per-cell verdicts.
* Full-campaign results doc + campaign1_summary.json once the last four
  boots land.
