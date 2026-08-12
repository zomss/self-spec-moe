# X13 — the batch crossover, and what a large window really costs

Date: 2026-08-13
Status: complete. **Both results overturn earlier statements in this phase.**

## Part A — the crossover has arrived, at batch ~32

R1, window 256, Marlin, K held at 4 across the range (schedule widened to
`[[1, 128, 4]]`; batch 32 re-run under the widened schedule as a control).

| batch | piecewise | whole-chain | ratio | device gain | bubble gain |
| --- | --- | --- | --- | --- | --- |
| 32 | 32.71 ms | 32.32 | **1.012x** | −4.75 | **+4.36** |
| 48 | 32.62 | 32.57 | **1.002x** | −4.73 | **+4.67** |
| 64 | 32.62 | 32.58 | **1.001x** | −4.70 | **+4.66** |

**Beyond batch 32 the two runtimes are indistinguishable.** Whole-chain's device
advantage (−4.7 ms, constant, as X8 found) is now exactly cancelled by a bubble
disadvantage of the same size. The prediction registered in X8 — that
whole-chain loses once the bubble gain exceeds ~4 ms — is confirmed, and it
happens right at the edge of the K=4 range.

**The control was necessary and it moved the number.** X8 measured b32 at
1.070x under the narrow schedule `[[1, 32, 4]]`; the same cell under
`[[1, 128, 4]]` measures 1.012x, with BOTH arms faster (piecewise 37.84 → 32.71,
whole-chain 35.35 → 32.32). Batch 32 sat exactly on the old schedule's upper
bound. X8's b32 point was therefore a boundary artifact, and the real curve is
flatter than X8 suggested — the margin was already nearly gone at 32.

## Part B — the scratchpad does NOT scale with window; the attention does

Whole-chain, per-step device time by family:

| R1 b=1 | binds? | wall | device | scratchpad | splitkv |
| --- | --- | --- | --- | --- | --- |
| w128 | yes | 22.27 | 19.15 | 0.528 | 1.071 |
| w256 | yes | 22.29 | 19.17 | 0.526 | 1.097 |
| w512 | no | 22.43 | 19.23 | 0.528 | 1.157 |
| w1024 | no | 22.58 | 19.52 | 0.526 | 1.370 |

| R5 b=8 | binds? | wall | device | scratchpad | splitkv |
| --- | --- | --- | --- | --- | --- |
| w128 | yes | 29.90 | 26.32 | 1.130 | 1.247 |
| w256 | yes | 30.22 | 26.54 | 1.129 | 1.483 |
| w512 | yes | 30.52 | 26.95 | 1.127 | 1.891 |
| w1024 | yes | 31.18 | 27.55 | 1.130 | 2.482 |

**The scratchpad copy is flat across an 8x window range** — 0.526–0.528 ms at
batch 1, 1.127–1.130 at batch 8. It is a fixed cost that scales with batch, not
with window.

What scales with window is the **splitkv attention kernel**, which is physical
and expected:

```text
R1 b=1:  splitkv = 1023 us + 333 ns per key-row
R5 b=8:  splitkv = 1048 us + 172 ns per key-row
```

### Correction

This phase earlier estimated materialisation at "~0.3 ns per key-row → 4.30 ms
at ctx 14336 batch 1, 34.41 ms at batch 8". **That fit was wrong**: it was
derived from two points that differed in BATCH (R1 b1 vs R5 b8) and attributed
the batch effect to the window. Measuring four windows within each batch shows
the window slope of the scratchpad is zero.

The concern was right but located in the wrong kernel. Extrapolating the
measured splitkv slopes to a non-binding window at ctx 14336:

| | attention cost of a non-binding window |
| --- | --- |
| batch 1 | ~5.8 ms/step |
| batch 8 | ~20.8 ms/step |

Against a whole-chain win of ~10 ms/step at batch 1 and ~0 at batch ≥32.

**These are 14x extrapolations beyond the largest measured window.**
`W98_WINDOWS` caps at 1024 and the boot contract enforces it; that contract was
not relaxed for a diagnostic. The slopes are fitted over four points spanning 8x
and are well determined within that range, but a knee beyond 1024 keys would not
be visible here.

## What this means for "whole-chain everywhere"

Combining both parts, the design is regime-dependent and the regimes are clean:

| regime | verdict |
| --- | --- |
| small batch, windowed | **whole-chain**, ~1.4x — the large win |
| batch ≥ 32 | **either** — 1.001–1.012x, the choice does not matter |
| long context, full attention | **piecewise + paged FA3** — a non-binding window costs 5.8–20.8 ms/step of extra attention, more than whole-chain returns |

So "whole-chain for all cases" is **unnecessary rather than impossible**. At
large batch it buys nothing; at long-context full attention it is actively
worse, because a non-binding window makes the draft read every key through a
kernel chosen for short key sets, which is exactly what paged FA3 is better at.

Route 1 from the earlier design note — replace `window: off` with a non-binding
large window — **holds only at short context** (≲1024 keys, where the added
attention is ~0.3 ms). At long context it is counterproductive.

## Consequence for the RL-rollout setting

EfficientRollout targets RL rollouts, which are batch-heavy. At batch ≥32 the
runtime choice is worth ~1%, so the whole-chain work does not pay there. It pays
at small batch — low-latency single-stream serving — which is a different
deployment than the paper's.

That does not retract the whole-chain result: the quant×skip8 anomaly was real,
host starvation was its cause, and at batch 1 whole-chain still returns 1.4x.
It bounds where the result applies.
