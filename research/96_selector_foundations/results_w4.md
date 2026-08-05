# W4 — completing the throughput model: transition, parked, and audited R

Data: `data/w4/`. Protocol: w3_preregistration.md (notune, ITERS=4,
episode rejection, GATE_DEBUG duty accounting). Runners:
`run_w4a_probe.sh` (transition + parked), `run_w4b_audit.sh` (llama cell
re-audit).

## W4a — transition cost vs duty cost (llama w512, R8 parks / R4 control)

Arm results (R8 = the parked cell; anchor = off):

| arm | R8 toks | vs off | measured duty (kpick) | measured arm/disarm flips |
|---|---|---|---|---|
| off | 2156.0 | — | — | — |
| parked (options stripped) | 2142.9 | **−0.6%** | 0.0% | 0 |
| dutyA (probe 2/64) | 2057.8 | −4.6% | 4.0% | 311 / 9837 steps |
| dutyB (probe 8/256) | 2071.7 | −3.9% | 3.2% | 79 / 9997 steps |

R4 control: off 600.9; parked 601.4 (+0.1%); dutyA 611.5 / dutyB 611.8
(+1.8% both) — probes did not degrade the winning cell.

Two-point solve of `loss = a·duty + c·flip_rate` with MEASURED duty and
flips (not the nominal schedule):

- **a ≈ 1.06 step-equivalents per armed step**
- **c ≈ 0** (slightly negative — noise)

### Verdicts

- **P-W4a2 REFUTED, and with it phase-95's ~20 ms/cycle transition
  story.** The per-flip arm/disarm cost is statistically ZERO. A 4×
  reduction in flip count at constant duty (dutyB) recovered only the
  duty difference. The loss phase-95 attributed to transitions is the
  physics of STANDING in a losing configuration: an armed step at a cell
  with R≈0.5, K∈{2,4} takes ~(KR+1)≈2× an AR step while its drafts are
  mostly rejected — ~1 step-equivalent lost per armed step, exactly the
  measured `a`.
- **P-W4a1 partially confirmed**: parked cost is real but smaller than
  the lottery-era estimate — **−0.6% at b16, ~0 at b8** (predicted
  −1..−3% / ~0). Model term: OFF's score is `1 − parked(cell)`, not the
  hardcoded 1.0.
- **C-C is already met for K/OFF switching.** The W0 audit said
  pre-captured verify widths make K-switching graph-free; this measures
  it end-to-end in deployment: flips are free. C-C's remaining burden is
  ONLY the window/skip graph swap (W7b), and the scheduler's 2%
  hysteresis rent prices a cost that does not exist while under-pricing
  wrong-arm duty (≈1 step-equivalent per armed step at losing cells).
- Probe design consequence (W8): probe cost = duty × ~1 step. The lever
  is duty, i.e. FEWER probing steps (the bandit's 4/256 = 1.6% halves
  the argmax path's 3.1%), not batching flips.

## W4b — llama compile-cell re-audit under notune

Pre-registered prediction (results_w2.md F6): audited R ≈ table R / 1.46
uniformly across (window, K, batch, ctx); non-uniform inflation
regenerates the whole llama table.

### Result: P-W4b REFUTED — the tables were never lottery-inflated

32 cells, deployed config × {w512, w2048} × {K2, K4}, notune, own off
anchor (`score_w4b.py`): **inflation mean 1.00×, range [0.91, 1.13]**.
The compile-cell R reproduces under notune. (Why the oracle escaped the
lottery: each oracle config ran under its own `VLLM_CACHE_ROOT`, freezing
one kernel draw per config; those draws were evidently equivalent to the
notune defaults on the compile-measurement path.)

**Correction to results_w2.md F6 (retained there per T11):** the "1.46×
inflation" was a CURRENCY artifact, not kernel inflation. The
"identity-derived true R" divided live accept by the SERVING S
(prefill-diluted, concurrency-shaped) and compared it against a
COMPILE-protocol R. Wrong denominator; the kernels were innocent.

### The real finding: a protocol gap in R, and an f gap by regime

At the same nominal cell (b8/c14k, K4, notune):

- **Compile protocol**: R ≈ 0.54–0.55 (both windows), accept 3.74–4.18.
- **Live serving** (W2d uncond): realized decode-phase S ≈ 1.6 at accept
  3.46–3.77 ⇒ implied R ≈ **0.33–0.43** — the deployment wins ~35% MORE
  than the map predicts there. The largest audit deviations (1.09–1.13)
  also sit at the c14000 cells.
- At R4 the direction reverses for a different reason: compile-content
  (packed C4) acceptance is **4.11** where live summarization content
  accepts **2.30** — an **f mismatch**, exactly C2's "f is regime-only"
  doctrine violated by using C4 as the universal compile content.

So the map's transfer error decomposes into two named, measured causes:
**R: compile-vs-serving protocol gap (map UNDERSTATES long-ctx serving
wins). f: compile-content vs regime-content gap (map OVERSTATES wherever
real content accepts less than C4).**

### W4c (open, next): protocol discriminator

One boot, both measurements: the compile-cell protocol (T(1+N)−T(1)) AND
a serving window at the same (batch, ctx) inside the same engine, so the
R gap is isolated from boot/content/kernel variance. Until W4c lands,
Round 1 (W5) treats compile-R as a SHORTLISTING signal only — consistent
with S1's "interpolation may shortlist, never decide."

## W4 model terms settled

| term | value | replaces |
|---|---|---|
| per-flip transition | ~0 | "~20 ms/cycle", the 2% hysteresis rent |
| wrong-arm duty | ~1.06 step-equiv per armed step | unpriced |
| parked engine | −0.6% b16, ~0 b8 | hardcoded OFF=1.0 |
| R source | compile-R shortlists; serving-R (W4c) decides | the lottery-audit question, closed |
| f source | per-regime measurement, never compile content | C4-universal f |
