# E5 — the LO lattice under the registered protocol: composed levers beat the single-lever baseline by 38%

Date: 2026-08-23. Twenty quantized arms plus `off` and `stock`, LO, batch 8,
**n = 32 submitted** (Phase 100's registered `4 x batch`), natural EOS,
instrument-free. Data: `data/e5_lattice_lo_b8_n32`. Scorer:
`scripts/score_e5_ceiling.py`.

E4 invalidated section 45's lattice: it ran `n = batch` unbackfilled, which
E3 measured carrying +-23% draw noise. This re-measures LO under the
protocol E4 showed has sd 3.6%.

## The ranking

| arm | vs stock | | arm | vs stock |
| --- | --- | --- | --- | --- |
| **`w256/skip8`** | **1.5795** | | `w128/skip4` | 1.4439 |
| `w1024/skip8` | 1.5389 | | `w128/skip0` | 1.3734 |
| `w512/skip0` | 1.5185 | | `w128/skip8` | 1.3306 |
| `w512/skip8` | 1.5074 | | `woff/skip4` | 1.1625 |
| `w1024/skip4` | 1.5026 | | `woff/skip8` | 1.1604 |
| `w256/skip0` | 1.4843 | | **`woff/skip0`** | **1.1444** |
| `w512/skip4` | 1.4741 | | `w512/skip16` | 0.9103 |
| `w256/skip4` | 1.4724 | | `w1024/skip16` | 0.9024 |
| `w1024/skip0` | 1.4681 | | `w128/skip16` | 0.8806 |
| | | | `w256/skip16` | 0.8385 |
| | | | `woff/skip16` | 0.8091 |

**Three tiers, cleanly separated.** Every windowed arm at skip 0/4/8 lands
1.33-1.58. Every unwindowed arm lands 1.14-1.16. Every skip16 arm lands
0.81-0.91, below stock. There is **no overlap** between the windowed and
unwindowed tiers.

## The headline

`woff/skip0` in the quantized family **is** EfficientRollout's lever: a W4
weight-only drafter with no other lever applied.

| | vs stock |
| --- | --- |
| best composed arm (`w256/skip8`) | **1.5795** |
| W4-only (`woff/skip0`) | 1.1444 |
| **composed over single-lever** | **+38.0%** |
| `off` (our runtime parked) | 0.8828 |
| spread best/worst | **+95.2%** |

**This is the byte-budget inversion measured end to end.** Phase 75's README
predicted it from their own Table 3: attention is **3.2%** of decode latency
at their (8K, b1) operating point, so a weight-only lever is the right
choice there. At (32K-capped, b8) the KV read is the dominant term, and a
weight-only lever addresses a bottleneck that is no longer binding. The
window axis -- which their method does not have, and which their section B.2
evaluates and rejects at *their* operating point -- is worth 38% at ours.

Both results stand. The law is the one Phase 75 stated: *a draft-only lever
wins iff it cuts the term that binds, and which term binds is set by
(dense|MoE) x (context x batch).*

## What the protocol fix changed

| | section 45 (n=8) | **E5 (n=32)** |
| --- | --- | --- |
| stock | 459.3 tok/s | **625.2** (+36.1%) |
| best arm | `w1024/skip8` 1.6595 | **`w256/skip8` 1.5795** |
| arms whose rank changed | — | **7 of 9 shared** |

Backfill alone raises the *baseline* by 36%, and the winner moves. Section
45's numbers are not comparable to these and are superseded at this cell.

## Two structural findings

**skip16 is below stock at every window** (0.81-0.91). D2(b) located the
additivity inversion at k=16 and the acceptance curves show why: tau collapses
to 1.9-2.2 against 3.6-4.5 at skip8. It is not a usable lever at any window,
which retires a quarter of the lattice.

**The window dominates skip at this cell.** The best arm pairs the *tightest
useful* window with the deepest usable skip (`w256/skip8`), and every window
beats no window regardless of skip depth. At 32K contexts the KV term is
large enough that cutting it outranks everything else.

## What this does NOT establish

* **The selector's value at a single point is zero by construction.** A
  per-cell selector has nothing to choose when there is one cell. The number
  that matters here is the arm SPREAD (+95.2%), which bounds what a wrong
  pick costs, and the margin over the single-lever baseline (+38.0%), which
  is what the extra lever axes buy. Neither is a switching result.
* **The comparison is our implementation of their lever**, in our runtime,
  not their system end to end. Their published 1.21x is at b1/2K, T=1.0, on
  A100; ours is b8, 32K cap, T=0, on H100.
* **Single boots at one draw.** E4 puts the draw noise at sd 3.6% for a
  pairwise ratio, so the tier separation (38%, 95%) is far outside it but the
  ordering *within* a tier (e.g. 1.5795 vs 1.5389) is not resolved.
* **Greedy.** Every number here is T=0; RL samples at 1.0.

## Scope

One cell, one batch, one prompt draw, quantized family. The 21
`target-matching` arms in the same directory are a bf16 half-lattice measured
under the same protocol by an earlier wrongly scoped launch; the scorer filters
on weight version and they do not enter this ranking.
