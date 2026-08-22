# E3 — the skip8 anomaly is prompt-draw luck, and it puts a 49-point band on every natural-EOS number

Date: 2026-08-23. Eight boots: `w1024/skip4` and `w1024/skip8` on **four
disjoint slices** of the frozen LO bundle (prompts 0-7, 8-15, 16-23, 24-31),
natural EOS, batch 8, traces on. Data: `data/e3_drain_off{0,8,16,24}`.

## The question

`w1024/skip8` measures best of twenty in the section-45 natural-EOS LO b8
grid (1.6595, replicated to 0.21%) and loses every fixed-batch and per-bucket
comparison. Six hypotheses were refuted. The seventh identified the
mechanism: throughput per step is `mean_active x tau`, and skip8 trades 7.8%
of acceptance for 13% more active requests. What remained untested was
whether that occupancy advantage is **arm-intrinsic** or **the luck of one
prompt draw**.

## The measurement

| prompts | occupancy Δ | tau Δ | tokens/step Δ | **throughput Δ** |
| --- | --- | --- | --- | --- |
| 0-7 | −0.6% | −9.3% | −9.8% | **−4.7%** |
| 8-15 | **+24.7%** | −5.5% | +17.9% | **+21.8%** |
| 16-23 | **+30.1%** | −5.3% | +23.2% | **+31.3%** |
| 24-31 | **−20.1%** | −9.4% | −27.6% | **−17.9%** |

(Δ = skip8 relative to skip4.)

| quantity | mean | sd | range | sign flips |
| --- | --- | --- | --- | --- |
| tau (acceptance) | −7.4% | **2.3** | [−9.4, −5.3] | **no** |
| occupancy | +8.5% | **23.3** | [−20.1, +30.1] | **yes** |
| throughput | +7.6% | **22.8** | [−17.9, +31.3] | **yes** |

## What this establishes

**Acceptance is arm-intrinsic. Occupancy is not.** `tau` puts skip4 ahead by
5-9% on every draw, sd 2.3 — a stable property of the arm. Occupancy swings
from −20% to +30% with sd 23.3 and flips sign twice. Throughput follows
occupancy, not acceptance.

**A single draw of eight prompts places skip8 against skip4 anywhere in a
49-point band.** Section 45's LO b8 grid is one draw. On slice 0-7 skip8 wins
the rollout; on slice 24-31 it loses by 17.9%.

**Replicating boots cannot detect this.** Section 44 replicated the grid and
found armed arms reproducing to 1.31% — but a replicate reuses the same
prompts, so it measures boot-to-boot noise and is structurally blind to
draw-to-draw variance. The two numbers answer different questions and only
one of them was ever asked.

## Why the drain does this

Under natural EOS each request stops at its own EOS, and the arms are
distribution-preserving but not bit-identical (section 21: batch-composition
numerics flip near-tie argmaxes). So each arm stops each request somewhere
slightly different, and with only eight requests the resulting drain shape is
dominated by a few stragglers. Occupancy is the time-average of that shape.
Under **equal work** every request generates the same length, occupancy is
constant by construction, and the comparison is clean — which is exactly why
equal-work and natural-EOS rankings disagreed throughout this investigation.

## What is damaged

**Section 45's per-point winners at long-output cells.** The lattice is one
draw of eight prompts per cell; at LO, where lengths span 4.6K-32.8K, the
draw noise on a pairwise comparison is ±20-30%. The derived claims inherit
it: "seven distinct winners over fourteen points", the "+7.28% equal-mix
ceiling", and the "+19.8% drain-shaped mix gaps" are all computed from
single-draw winners.

**Not damaged:** everything measured at equal work — the cost fits, the
band tables (E1d), the batch sweeps (E1e), section 25's equal-work lattice
comparison, and section 49's K sweep. Acceptance curves are also unaffected:
`tau` is stable across draws.

## What it changes about the method

**Long-output natural-EOS comparisons need multiple prompt draws, not
multiple boots.** For a cell whose lengths span 7x, eight requests is too few
for the drain to average out. Either raise n per cell (Campaign 1's
`n = max(16, 2b)` is a start but likely still short at LO), or report over
draws with a spread, or compare at equal work and treat natural EOS as a
separate, noisier deployment measurement.

**And occupancy is a real term, just not a lever.** `mean_active` genuinely
drives throughput — it moved 30% here — but it is a property of the draw
crossed with the arm's divergence, not something a selector can choose. It
belongs in the error bars, not in the model.

## Scope

n = 4 draws, one cell, one batch, one arm pair. The variance is estimated
from four points, so the sd figures are indicative rather than tight; what is
not in doubt is the sign flipping, which needs only two draws to establish.
