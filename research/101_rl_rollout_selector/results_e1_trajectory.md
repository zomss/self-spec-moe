# E1 — the trajectory priced: intra-rollout lever switching is worth under 0.5%

Date: 2026-08-22. Analysis over phase-98 measured data; no boots.
Script: `scripts/e1_trajectory.py` (+ `e1_variants.py`), output
`data/e1_trajectory.json`.

## Setup

The continuous model of `design_round1_continuous.md`, instantiated with:
the two-batch long-context cost fit, the verify split solved on the parked
arm, tau(u) from the measured LO depth-8 prefix profiles (PCHIP over bucket
centres, flat past 8K), and `B(u)` from the **measured natural-EOS length
distribution** of the LO b32 stock run (32 requests, 4.6K-32.8K tokens,
median 13.1K). Fifteen arms (quantized family, skip16 out), K=4. The
omniscient schedule comes from DP over a 64-token u-grid with a per-switch
stall of the whole batch.

**E1a — the integrator ranks credibly.** Fixed-arm totals from this exact
integrator against the measured b32 grid: Spearman **0.825**, and the
measured best arm sits in the model's top-3. So the flat result below is not
an artifact of a broken model.

## E1b — the result

| switch cost | best static | schedule | **gain** | switches |
| --- | --- | --- | --- | --- |
| 0 ms | `w512/skip4` | — | **+0.39%** | 6 |
| 50 ms | `w512/skip4` | — | +0.32% | 4 |
| 500 ms | `w512/skip4` | — | +0.01% | 1 |
| 5 s | `w512/skip4` | — | 0.00% | 0 |

Sensitivity, at 50 ms switch cost:

| scenario | gain |
| --- | --- |
| base: B0=32, median 13K | +0.32% |
| outputs x2 (median 26K) | +0.24% |
| outputs x4 (median 52K) | +0.21% |
| B0=128 | +0.29% |
| B0=128 and outputs x2 | +0.22% |

**The gain is flat in every direction that was supposed to grow it.** Longer
context and larger batch do not help; they mildly hurt, because the
trajectory spends even more of its time deep in one arm's region.

## Why, and why this is not a modelling artifact

The schedule the optimizer finds is exactly the 1.2 hypothesis --
`woff/skip0` for the first ~600 tokens, tightening windows through 1-4K,
`w512/skip4` for the long middle, `w1024/skip4` once the batch drains below
~10. **The physics is right. The integral is small**, because within one
content cell the top arms are near-ties everywhere along the curve: the
measured b8 grid itself has the top three arms within 2.7% of each other,
and the crossings all happen in the first ~4K tokens of a 13K-median
rollout. The +19.8% gaps that motivated the direction are **cross-content**
(LI-vs-LO pairs), and a rollout trajectory never crosses content -- it only
moves (u, B).

This is the dwell-time law's **fifth** appearance (+1.4% per-regime, +0.15%
per-cell, +2.37% swept-batch, 83%-captured, and now +0.4% intra-rollout),
and its cleanest: even an omniscient schedule with free switching cannot
beat the best static by more than ~0.4% along this trajectory.

## The registered criterion fires

`README.md` registered: **gain under ~3% fails the switching thesis on this
workload family.** It fires at 0.4%. Intra-rollout lever switching, at fixed
K, on this lattice and model, is not a mechanism that can meet objective 1
(+10% over the best static).

## What survives, and where +10% would have to come from

1. **Cross-content batch partitioning.** The measured 19.8% gaps are between
   content families at different batches. A real RL iteration that mixes
   content (math + code + dialogue) can assign arms **per request group**
   rather than per u-segment -- selection across groups, not switching along
   a trajectory. This is the mechanism our own data supports, and it needs no
   new physics: it is phase 98's per-cell selection applied within one
   iteration.
2. **The K axis, jointly.** E1 held K=4. The measured K sweep shows per-arm
   K spreads of 12-16%, and section 49 shows the analytic K model cannot
   place the optimum -- so pricing a (arm, K) schedule needs a K-augmented
   grid, not this integral.
3. **T=1 (E2).** All margins here are greedy. Sampling could compress or
   widen them; nothing here predicts which.
4. **Other models** (objective 3): the near-tie structure of the top arms is
   a property of this model's acceptance curves; a model with steeper
   acceptance falloff would have larger crossings.

## Scope

Model-space comparison under one model, so level error largely cancels;
ranking validated at 0.825. The x2/x4 length scenarios extrapolate the cost
model past its 13K calibration context (the s38 evidence says long-end
calibration transfers down, and LO's own 32K contexts already ranked with a
top-two swap). Lockstep-u approximation inherited from phase 98. T=0.
