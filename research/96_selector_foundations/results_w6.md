# W6 item 2 — dense Round-2 calibration: the gate passes on measurement

Data: `data/w6/w6cal_*.json`, scorer `scripts/score_w6_calib.py`.
Protocol: unconditional K4, q-hum pool {skip 2,8 | none} × {w512 |
w2048}, regimes R5/R5cot/R1, seeds 0+1 (disjoint content), piecewise,
notune, ITERS=4, episode rejection, per-seed AR anchors (anchor
certification: seed-0 values match W2 to 0.2–1.0%).

## Measured (S_dec, seed-mean; f per seed in the scored JSON)

| comp | R5 | R5cot | R1 |
|---|---|---|---|
| s-2,8 / w512 | 1.278† | **1.884** | 1.161 |
| s-2,8 / w2048 | 1.442 | 1.873 | **1.184** |
| s-none / w512 | **1.532** | 1.857 | 1.125 |
| s-none / w2048 | 1.454 | 1.827 | 1.148 |

† seed-1 S collapsed (1.073 vs seed-0 1.483) at healthy accept —
uncertified per protocol (suspected long episode); does not decide any
ranking (this comp wins only R5cot, where its seeds agree 1.935/1.833).

- **P-W6b CONFIRMED**: f agrees across disjoint content seeds within
  0.5–1.5% at most cells (worst 6% at R5cot/w512 — the content band).
- **P-W6a CONFIRMED**: intervals shrink to seed-spread; every regime is
  now rankable. The uncond values also expose policy-path losses on
  dense (R5cot measured 1.88 vs policy-run 1.68).

## Gate arithmetic (W3 thresholds, 6-regime basis; R4/R8/R6 are
{OFF}-tied for T and B by the Round-1 tie-sets, contributing 0)

Best static: **s-none/w512** (3-regime mean 1.505).

| regime | T (per-regime best) | B (static) | Δ |
|---|---|---|---|
| R5 | 1.532 (= static) | 1.532 | 0 |
| R5cot | 1.884 (s-2,8/w512) | 1.857 | +2.8% |
| R1 | 1.184 (s-2,8/w2048) | 1.125 | **+5.9%** |

6-regime mean(T−B) = +1.4% (< 2%); best single regime = **+5.9% ≥ 5%
with mean ≥ 0 → PASS via the single-regime branch — this time on
MEASURED values with certified anchors and agreeing seeds**, unlike
W5's miscalibrated letter-PASS.

## Verdict

- **Dense: the per-regime selector is justified.** The winning ladder:
  R5 → s-none/w512; R5cot → s-2,8/w512; R1 → s-2,8/w2048; R4/R8/R6 →
  OFF. Note the R1 winner needs a different SKIP than the R5 winner —
  a boot-class switch — so the deployment either picks the two-boot
  portfolio or accepts B (still 1.505 mean) with the single boot;
  the portfolio decision is the two-stage knapsack instance (w6_design
  §3) with real numbers.
- **Llama: unchanged — static q-w4a16_w-2048_s-b2 + OFF gate.**
- Consistent with the error floor: the offline-only search would have
  picked w2048 at dense R5 (−3.6%) and missed R1's skip choice; Round 2
  recovered both for ~1 GPU-h.

## Skip-identity subset transfer test (4 single-layer probes × R5/R1 × 2 seeds)

Sensitivity = f(none) − f(skip-L), seed pairs agree (cross-seed τ = 1.00
both regimes):

| layer | R5 (s0/s1) | R1 (s0/s1) |
|---|---|---|
| 2 | 0.009 / 0.012 | 0.012 / 0.029 |
| 8 | 0.038 / 0.033 | 0.011 / 0.006 |
| 18 | 0.067 / 0.055 | 0.026 / 0.053 |
| 30 | 0.182 / 0.158 | 0.103 / 0.072 |

Orders: R5 [2,8,18,30], R1 [8,2,18,30] — exactly ONE adjacent swap;
cross-regime τ = 2/3. The pre-registered rule said "at most one adjacent
swap (τ ≥ 0.67)": the verbal clause PASSES, the numeric threshold fails
by rounding (0.667 < 0.67). Both readings reported; the boundary case is
ours, not the data's.

The substantive finding is sharper than the formal verdict: the
expensive tail (L18, L30; depth-monotone) transfers perfectly, but the
CHEAP tier's internal order (L2 vs L8) genuinely flips by content class
with seed-consistent effect sizes (R5: L8 costs 3× L2; R1: half).
**The knapsack's decisive entries — which cheap layers to pick at small
counts — are precisely the non-transferable ones.** The dichotomy,
fractally.

Design consequence: one-time per-arch profile → identifies the cheap
tier (transferable); Round 2 selects WITHIN the cheap tier per regime
(a handful of bursts, not L boots). G3's gated measurement mode drops
off the critical path.

## 36-layer profile + count-{4,8} ladder (seed-mean, w512)

| count | R5 f / e2e | R1 f / e2e |
|---|---|---|
| 0 | 0.857 / 523 | 0.952 / 169 |
| 2 {2,8} | 0.813 / 450† | 0.935 / 174 |
| 4 {2,7,8,11} | 0.740 / 499 | 0.924 / 167 |
| 8 {2,3,4,7,8,11,15,16} | 0.612 / 484 | 0.841 / **192** |

† episode-contaminated seed-1 (uncertified; certified ≈505).

Full profile (36/36): cheap tier is exclusively early-mid layers —
depth-monotone at full resolution. **Count-8 is the new R1 winner:
e2e 192 ≈ S 1.25 vs AR, +9% over the previous best (s-2,8/w2048, 176)**
— deep skipping pays at b1 math (f 0.84 still clears the halved cost)
and loses at b8 RAG (f falls faster than cost). The count axis is
regime-dependent exactly as the window axis was: one more per-regime
lever for the ladder, and the W6 gate's R1 margin widens from +5.9%
to ~+15% with count in the pool.
