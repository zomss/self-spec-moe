# W11 results — llama + q3_32b under equal work, and a structural finding

Pre-registration `w11_llama_q32_remeasure.md` (91c76bc32, before data).
Scorer `scripts/score_w11.py` → `data/w11/w11_scored.json`.
11 boots, all complete, notune, per-regime fixed length, ITERS=4.

## Counts

| arch | Stage B | W11 | predicted | winning configs |
|---|---|---|---|---|
| llama | 24/33 | **20/33** | 17–22 (point 20) | 5 → **4** |
| q3_32b | 25/33 | **15/33** | 14–22 (point 19) | 3 → **1** |

## Prediction outcomes

- **P-W11a CONFIRMED exactly**: llama 20/33, the registered point.
- **P-W11b CONFIRMED**: q3_32b 15/33, inside 14–22 but at the low end.
- **P-W11c CONFIRMED**: q3_32b lost proportionally more — it retained
  60% of its wins (15/25) against llama's 83% (20/24), as predicted
  from its compressed margins. This was the deflation-independent
  comparative test.
- **P-W11d SPLIT — CONFIRMED for llama (4 configs), REFUTED for
  q3_32b (1 config).** Only `w4gptq_k4` wins any q3_32b cell;
  `w4gptq_k6`, `win512_k4` and `skipb2_k4` win nothing. See below for
  what this does and does not cost C1.
- **P-W11e CONFIRMED on both**: llama 1 flip up (R6 b8), q3_32b 2
  (R5cot b8, R8 b1), against 5 and 12 wins fallen respectively.

## The anchor term is architecture-specific — MoE was the outlier

b1 cells are drain-immune by construction, so their shift isolates the
autotune/protocol term:

| arch | b1 anchor shift (mean) |
|---|---|
| MoE (W10) | **+3 to +14%** |
| llama | +1.2% |
| q3_32b | **−1.2%** |

I previously described the lottery-slowed anchor as
"architecture-general". That is right in KIND and wrong in MAGNITUDE:
only MoE's Stage-B anchor was materially slow. llama's and q3_32b's
were fine, so **their corrections are essentially pure drain**. The
correction stated in `results_w10.md` §"Consequences" is narrowed
accordingly.

The separation is visible cell-by-cell. q3_32b b1 cells move +0.2% to
+3.9%; its b32/b64 cells move −11.7% to −42.2%. Drain-immune cells
hold, drain-exposed cells collapse.

## The structural finding: dense and sparse have OPPOSITE batch-dependence

Every one of q3_32b's 15 surviving wins is at **b1 or b8**. It loses
at *every* b32 and b64 cell. llama shows the same direction (wins
through b32, loses at every b64 cell). Stage B's natural-EOS numbers
had suggested wins extending to high batch on both; under equal work
they do not.

Set against W10/W7, the four architectures split cleanly:

| family | low batch | high batch |
|---|---|---|
| dense (llama-8B, Qwen3-32B) | **wins** | loses |
| sparse (MLA V2-Lite, MoE 30B-A3B) | loses (acceptance-independently) | marginal wins only |

W8's cost model explains both directions with one mechanism. For a
dense target, the b1 step is memory-bound, so a W4 draft reading ¼ the
bytes is genuinely cheaper and speculation pays; at high batch the
target step turns compute-bound, the draft's byte saving stops
converting, and verify's extra tokens cost real time. For a sparse
target the b1 step is *dispatch*-bound (measured 7.8–15.2× off its
byte roofline), so quantization buys nothing and no acceptance can
rescue it; only at b32, where expert coverage saturates and the step
becomes bandwidth-bound, does the draft's byte saving finally convert.

**Speculation pays where the target step is bandwidth-bound — and the
batch at which that happens is opposite for dense and sparse
architectures.** That is a sharper statement of C1's regime axis than
"the winner changes with batch", and it is now mechanistically grounded
rather than observed.

## What this costs C1 — and what it does not

C1's written claim, *"No configuration wins across any single
architecture's surface"*, **still holds for q3_32b**: 18 of its 33
cells are OFF, so no config wins the surface. (MLA already had a
single winning config in c1 under the same reading.) The
cross-architecture claim is untouched — five architectures still show
five distinct winner profiles.

What changes is q3_32b's *within-column richness*: c1 described it as
"quant-dominant; window/skip at scale", and window/skip now win
nothing. The honest row is **one winning e2e config (W4-GPTQ K4),
wins confined to b1/b8**. My P-W11d was stricter than C1's own claim,
and it failed on this architecture; I record that rather than
retro-fitting the prediction.

## Disclosed confound (small, localized)

Fixed length raises peak KV pressure — no request finishes early — so
preemptions rise (q3_32b spec arm 32 vs Stage B's 12), concentrated in
three long-output b32 cells (R4, R5, R5cot). Of the 12 fallen q3_32b
wins, only **R5cot b32** is materially exposed (12 preemptions spec vs
8 off); the large drops (R8 b64 −42%, R1 b64 −35%, R8 b8 −33%) have
**zero** preemptions and are pure drain. b64 cells exist only for
short-output regimes, so none is preemption-affected.
