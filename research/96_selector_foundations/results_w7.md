# W7 results — MLA/MoE Round-2 completeness pass

Protocol and predictions: `w7_completeness.md`, committed at 76d51f60d
BEFORE any run existed. Artifacts: `data/w7/w7_{mla,moe}_{off,uncond}_*`
(8 boots, all notune-stamped, 12 cells each), scored by
`scripts/score_w7.py` → `data/w7/w7_scored.json`. MLA boots ran as
GPU0/GPU1 lanes (cross-boot certification across DIFFERENT GPU lanes);
MoE TP2 boots are sequential reboots on GPUs 0+1.

## Certified cell table (S = uncond/off, certified means)

| arch | rid | b1 | b8 | b32 | b64 (map-only) |
|------|-----|-----|-----|------|------|
| MLA | R1 | (0.60)* | 0.955 | **1.126 ARM** | 1.118 |
| MLA | R2 | 0.577 | 0.909 | **1.129 ARM**† | 1.085 |
| MLA | R6 | 0.568 | (0.93)* | **1.111 ARM** | 1.127 |
| MoE | R1 | 0.615 | 0.886 | (0.91/0.94 OFF)* | (0.90)* |
| MoE | R2 | 0.533 | 0.647 | ~~1.087 ARM~~ **RETRACTED→OFF** | 1.067‡ |
| MoE | R6 | 0.492 | 0.809 | 0.857 OFF | 0.914 |

\* uncertified under the 2% cross-boot rule but VERDICT-ROBUST: both
boot readings fall on the same side of 1.0 (values in parens =
conservative reading). † resolved by tie-breaker boot, see P-W7c.
‡ MoE b64 rests on the same drain-contaminated protocol as the
retracted b32 cell (W9); treat as unverified, not as a map row.

## Verdicts

**P-W7a (MLA below b32) — CONFIRMED.** Every certified cell S < 1
(0.57–0.96); the two uncertified cells are OFF under both readings.
Back-out R̂ = 1.06–1.95 against the banked 1.21–2.29 band. The strong
form held where measurable: τ ≈ 5.0 (the maximum) and the cells still
lose — acceptance-independence demonstrated live.

**P-W7b (MoE below b32) — CONFIRMED.** All six cells certified OFF,
S = 0.49–0.89, R̂ = 1.10–2.25. Same live demonstration: R6 b8 runs at
τ = 4.81 (f = 0.95) and still loses 19%.

**P-W7c (MLA b32) — ARM, with the pre-registered content flag.**
R1 S = 1.126 and R6 S = 1.111 certified, per-lane best-2 S ≥ 1.10 on
every lane (bar was 1.02). R2 uncond initially failed cross-boot
certification by 0.4pp (2802 vs 2735, a 2.4% lane offset); both lanes
independently clear the bar (per-lane S = 1.170 / 1.129), so the arm
verdict never hinged on the tie-break. Tie-breaker boot3 (GPU1,
`w7_mla_uncond_boot3_r2b32.json`): rounds [2707.3–2768.8], agrees with
lane1 to 0.06% (vs 2.34% against lane0) — lane0/GPU0 was the odd boot
out, HIGH this time (the lane effect is an offset, not a one-sided
suppression). Certified 2-of-3 pair: uncond 2736.0 / off 2423.1 →
**S = 1.1291**, ARM stands certified.
Flag: MLA b32 content is ceiling-clipped loops (clip 0.89–0.97,
out_p50 = 2048 = ceiling, τ ≈ 5.0 inflated). Per pre-registration this
ARM verdict requires real-content confirmation before deployment; the
OFF verdicts are a fortiori sound.

> **RETRACTED 2026-08-06 by W9 (`results_w9.md`).** The refutation
> below does not stand. Under a workload-controlled protocol the cell
> LOSES at both engine configurations (S = 0.928 / 0.943, τ unchanged);
> the apparent win was a drain artifact — the two arms generated
> different text (out p95 1332 for AR vs 925 for spec), so the AR arm
> spent longer in the inefficient low-batch tail. The oracle's OFF
> prediction for MoE b32 was correct. Text preserved per T11.

**P-W7d (MoE b32) — REFUTED in R2 (retained per T11).** Predicted OFF
everywhere from oracle S_ref 0.94–0.99; measured R2 b32 S = 1.087,
certified, per-boot 1.090/1.108, on NATURAL content (clip = 0.0,
out_p50 = 203, τ = 4.43, f = 0.86) with R̂ = 0.769 vs banked 0.951.
The oracle's R was pessimistic for this cell; real acceptance clears
the true break-even with ~9% to spare. R1 (0.91/0.94) and R6 (0.857)
b32 stay OFF as predicted — R6 despite τ = 4.84, because R̂ = 1.16:
the C2 dichotomy in one row (R decides, not acceptance). b64 R2
S = 1.067 is map-consistent with the arm.

**P-W7e (b64) — map rows recorded, no arming decisions.** MLA b64
S = 1.09–1.13 across all regimes; MoE R2 1.067, R6 0.914. MoE R1 b64
is anomalous but REPRODUCIBLE (off ≈ 955–980 in both boots, half of
b32): clip ≈ 5% shows ~3/64 requests hit the 16384 ceiling and
dominate the batch tail — a content property of R1 at this width, not
an episode; uncond additionally uncertifiable (round spread 519–979).

## Episode dossier addendum

- MLA R1 b1 uncond, GPU0 lane: rounds [99.2, 99.3, 137.4, 148.0] vs
  GPU1 lane [148.7–149.9] — two deep-episode rounds, GPU0 again.
- MLA R6 b8 off: clean within-lane spreads, 2.3% cross-lane offset
  with GPU0 low — the lane-offset signature.
- MoE R1 b32 off boot1: bimodal [1762, 1767, 2058, 2066] (TP2 spans
  both GPUs; not lane-localizable).

## What this closes

The two-round design now has Round-2 verdicts for all four
architectures: dense PASS (armed ladder), llama fail-closed OFF,
MLA gate-OFF below b32 + b32 arm (content-flagged), MoE gate-OFF
except a certified natural-content b32/R2 arm the oracle missed —
the last point is itself evidence for the design: Round 1's sound
elimination kept the cell alive, and only Round 2's measured
acceptance could bank it.

**Superseded by W9**: the MoE b32/R2 arm is retracted (drain artifact,
see the P-W7d note above). MoE's Round-2 verdict is gate-OFF
everywhere, matching Round 1's prediction; MLA b32's arm stands.
