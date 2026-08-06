# W9b — drain-contamination sweep of prior phases

Follow-up mandated by `results_w9.md` §4. Screen:
`scripts/sweep_drain.py` → `data/w9/drain_sweep.json`.

## Screening rule

A cell's e2e ratio is contaminated iff ALL hold:

1. **batch > 1** — `run_grid.py` runs b1 as four SEQUENTIAL single-
   request generates, so there is no drain and no tail asymmetry.
2. **not fully ceiling-clipped** — if every request hits the ceiling
   (clip = 1.0) all lengths are equal and no drain exists.
3. **the arms' length distributions diverged** — |p95_spec/p95_off − 1|
   > 10%. This is the direct W9 signature (1332 vs 925), and it is the
   *mechanism*, not a proxy: equal-length arms cannot produce the
   artifact no matter how heavy the tail.

A contaminated cell whose S is far from 1 is **unsound but
verdict-robust** (the ratio is wrong, the sign is not). Contaminated
AND 0.90 ≤ S ≤ 1.15 is **SUSPECT** — the artifact can flip it.

## Result: 140/726 Stage-B cells contaminated, 43 SUSPECT

| | count | share |
|---|---|---|
| Stage-B paired cells | 726 | — |
| b1-immune (sequential) | 198 | 27% |
| clip-immune (all ceiling) | 21 | 3% |
| **contaminated** | **140** | **19%** |
| **SUSPECT (near-1 + contaminated)** | **43** | **6%** |

Worst tail gaps are extreme: `stageb_llama_w4a16_k4` R1 b64 shows
p95 3529 (AR) vs 16384 (spec) — a 364% gap, i.e. the spec arm ran
requests to the ceiling that the AR arm terminated. Its S = 1.028 is
meaningless. Also flagged: `stageb_q3_32b_w4gptq_k4` R1 b64
(16384/8048, S = 1.137) and `stageb_moe_win8192_k3` R2 b32
(1332/923, S = 1.122).

Beyond Stage B the sweep re-flags the already-retracted W7 MoE R2 b32
(S = 1.085, gap 30.6%) — the screen reproduces the known case
unprompted, which is its validation — plus W7 MoE R6 b64 (S = 0.910,
gap 16.9%), already an OFF verdict and unchanged in sign.

## The mitigation: winner-map comparisons partly cancel

C1's winner map picks the best CONFIG per cell, i.e. a spec-vs-spec
comparison — and the shared AR anchor cancels exactly:
S_A/S_B = rate_A/rate_B. So the gate decision (spec vs OFF) carries
the artifact in full, while the winner-map decision only carries the
tail difference BETWEEN the competing spec arms. That residual is not
small:

| arch | arms | multi-request cells | >10% tail spread between arms |
|---|---|---|---|
| dense | 5 | 24 | 10 (42%) |
| llama | 7 | 24 | 13 (54%) |
| moe | 4 | 24 | 9 (38%) |
| q3_32b | 4 | 24 | 11 (46%) |
| **mla** | 2 | 24 | **0 (0%)** |

MLA is clean by construction — its ceiling protocol clips everything,
so all arms generate identical lengths. This is the same property that
made MLA's b32 arm reproduce across protocols in W9 while MoE's did
not, now confirmed as systematic rather than lucky.

## What is and is not at risk

**Not at risk.** (a) Acceptance measurements (τ, f) — unaffected, the
artifact is purely a wall-clock/length effect; (b) all b1 cells (27%);
(c) MLA everywhere; (d) contaminated cells far from 1 — 97 of the 140
keep their sign; (e) the phase-96 dense results, which used
`w6cal`-style fixed-window protocols, and the ladder validation, which
compared arms at matched decode lengths.

**At risk.** The 43 SUSPECT Stage-B cells, and any C2 oracle row
derived from them — this is precisely the class that produced the
retracted MoE b32 arm. The C2 tables invert S to R, so a contaminated
S yields a wrong R for that cell.

**Assessment.** The headline C1/C2 claims are dense-architecture wins
far from S = 1 and acceptance-independent OFF verdicts, neither of
which this artifact can move. What it does undermine is the *marginal*
cells — exactly where a selector spends its decisions. No published
headline is retracted by this sweep; the honest statement for the
paper is that marginal cost-side cells measured under natural EOS with
a draining batch require equal-work re-measurement before being
claimed, and that 43 such cells exist in Stage B.

## Follow-through (W10, done for MoE)

`results_w10.md` re-measured the whole MoE surface under equal work.
It confirms this screen's two-sided caution empirically (5 wins fell,
2 losses rose) and adds a SECOND inflation mechanism this screen does
not detect: Stage-B AR anchors ran 3-14% slow under autotune, and
since S = spec/AR a slow denominator inflates every S in the arm
uniformly — including drain-immune b1 cells. That mechanism is
architecture-general, so the counts in every c1 row (not just the
contaminated cells listed above) carry a few-percent anchor
uncertainty. Only MoE's changed qualitatively.

## Recommended remediation (not run)

Re-measure only the 43 SUSPECT cells under `G93_FIXED_LEN` — roughly
one boot per (arch, config) pair, since a fixed-length boot yields all
its regimes at once. Cheaper alternative for the paper: state the
verdicts for these cells as unresolved rather than re-running, since
none carries a headline.
