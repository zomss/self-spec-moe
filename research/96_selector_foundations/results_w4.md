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

### W4c — the gap is a REALIZATION × DECODE-LENGTH effect, not protocol arithmetic

Design: `run_w4c_protocol.py` — one engine, ladder {cmap = the audit's
byte-identical compile prompts, r5 = deployment content} × {compile-N160
(the map's exact min-min arithmetic), compile-N512, serving-512
(ignore_eos), serving-natural-EOS} × realization {scratchpad =
FULLCG+wholechain (the map's and phase-95's deployed-C3 stack, per
95/run_e0.sh), piecewise = paged window chain}. Matched off anchor.
Anchors: off/r5/serving 140.1 ≈ W2's 142.4; piecewise spec/r5/serving
**198.9 ≈ W2d's 198.7 (0.1%)** — the attribution is exact.

**Disclosure (instrument error, mine):** every 96 llama serving run
before W4c (W1 matrix, W1b, W2, W2b, W2c, W2d) exported the e3-gate env
set, which lacks `VLLM_SELF_SPEC_DRAFT_FULLCG/WHOLECHAIN` — so they ran
the PIECEWISE realization while phase-95's llama numbers and the map ran
the scratchpad. W1's variance causes, W2's currency findings, and W4a's
scheduler results are realization-independent claims and stand; but all
my llama serving S values are piecewise-stack values, and the
"map-vs-serving R gap" decomposes as follows.

R at llama b8/c14k K4 w2048 s-b2, notune (dec basis; S_e2e where noted):

| measurement | scratchpad | piecewise |
|---|---|---|
| compile-N160, cmap (the map's cell) | 0.562 | 0.561 |
| compile-N512, cmap | 0.506 | 0.465 |
| serving-512 dec, cmap | 0.533 | 0.527 |
| serving-512 dec, r5 | 0.504 | 0.494 |
| **natural-EOS serving, r5 (~72 tok/req)** | S_e2e **0.846**, accept 3.163 | S_e2e **1.420**, accept 3.765 |

Verdicts:

- **Protocol arithmetic is nearly innocent** (P-W4c1 axis: +5–6%
  within-realization, same content). The N-window bias is real but
  modest: N160 overstates R by ~6–17% vs 512-token serving (P-W4c2) —
  correctable by lengthening/reporting per-N R.
- **Long decodes: the realizations are equivalent** (within 2%
  everywhere at 512-token decodes) and the map's R is ~right.
- **Short decodes: the realization lever flips sign and is worth 68%.**
  At the deployment's actual R5 behavior (~72-token answers after 14k
  prefill), scratchpad LOSES 15% vs AR (and accepts less: early-cycle
  numerics, the phase-67 kernel-switch perturbation) while piecewise
  WINS 42%. The compile protocol is doubly blind to this cell: it forces
  `ignore_eos` AND its C4 content never stops.
- P-W4c3: EOS/decode-length matters (11% on R even within piecewise) —
  it is not a nuisance term, it is the axis.

**This closes the loop with I2**: the map's missing index (generated-
suffix length) carries not only the acceptance term phase 95 found but a
first-order COST/REALIZATION term. A regime carries its decode-length
distribution by construction — Round 2's per-regime confirmation is the
fix, now for two measured reasons.

Consequences:

1. **Realization enters the composition pool as a boot-class lever**
   (W0's matrix already classed it; now it has a measured 68% regime-
   dependent swing). The deployed C3 stack (scratchpad wholechain) is
   the WRONG realization for short-decode regimes; where the scratchpad
   retains a win region at all must be re-located with the corrected
   instrument.
2. Round 1's compile protocol gains a short-decode arm (small-N R
   reported per N), or short-decode regimes are excluded from compile
   coverage and owned entirely by Round 2.
3. Compile-R remains SHORTLIST-only (S1), now with its two measured
   blind spots named: N-amortization bias (~10%) and the short-decode
   realization cliff (~68%).

## W4 model terms settled

| term | value | replaces |
|---|---|---|
| per-flip transition | ~0 | "~20 ms/cycle", the 2% hysteresis rent |
| wrong-arm duty | ~1.06 step-equiv per armed step | unpriced |
| parked engine | −0.6% b16, ~0 b8 | hardcoded OFF=1.0 |
| R source | compile-R shortlists (blind spots: ~10% N-bias, 68% short-decode realization cliff); Round 2 decides | the lottery-audit question, closed |
| f source | per-regime measurement, never compile content | C4-universal f |
| realization | boot-class composition lever, regime-dependent sign (scratchpad −15% / piecewise +42% at R5-natural) | assumed fixed deployed stack |
