# G98-G full grid: the selector against omniscient, on one box

G98-G's reduced confirmation could only report `selector / best-in-set` over
a 10-cell candidate set, because a 31-cell grid compares cells ACROSS boots
and this box changes state between them. A drift measurement then showed the
comparison is safe after all, in one direction:

| regime | armed cells, round to round | OFF cell, round to round |
| --- | --- | --- |
| R1 | 1.44% | **14.96%** |
| R4 | 1.24% | **10.93%** |
| R5 | 1.01% | 9.52% |
| R5cot | 1.18% | 9.49% |
| R6 | 0.52% | **15.92%** |
| R8 | 1.32% | **15.11%** |

Armed cells repeat to ~1%; OFF moves 9-16%. That is the clamp's own
arithmetic — a fixed per-step host delay is enormous against an 8 ms parked
step and modest against a 40 ms armed one — so **OFF is the most
state-exposed cell in the grid, not the least**. Normalising by OFF was
tested and makes drift ~10x worse; it is not used.

So armed-vs-armed rankings are measurable here, and the full grid was run:
**31 cells x 6 regimes x 2 rounds with order reversed = 62 boots.**

## Result

| regime | selector pick | tok/s | omniscient (best of 31) | regret |
| --- | --- | --- | --- | --- |
| R1 | `woff/skip4` | 158.9 | `w512/skip4` | 1.76% |
| R4 | `woff/skip0` | 729.9 | `woff/skip4` | 1.30% |
| R5 | `w512/skip4` | 881.7 | `w512/skip0` | 3.33% |
| R5cot | `w512/skip4` | 1004.3 | `w512/skip4` | **0.00%** |
| R6 | `w256/skip8` | 3890.8 | `woff/skip4` | **10.59%** |
| R8 | `woff/skip4` | 2363.5 | `w512/skip4` | 0.22% |

Every pick is `w4a16-quantized`; only the window and skip vary, as in every
other arm of this phase.

**Equal-weight, time-weighted aggregate:**

| quantity | value |
| --- | --- |
| selector | 572.8 tok/s |
| omniscient | 583.3 tok/s |
| **selector / omniscient** | **0.9820** |
| selector / OFF | 1.415x (±10% band — OFF is state-exposed) |

D3 on h103 measured 0.951 of omniscient over the same lattice. This box
gives **0.982** from a selector fitted entirely on its own cost profiles,
with acceptance transferred from h104 and checked against realized values
(median ratio 0.977 over 54 pairs).

The R6 miss is the one already diagnosed (`results_g98_g_r6.md`): the
acceptance transfer is optimistic by +16-17% for deep-skip cells against
+3-6% for shallow ones, so the selector over-rates `skip8`. It survives here
at 10.59% and is worth ~0.23% of the aggregate, because R6 carries 2.2% of
the time in a token-weighted mix.

## Replication on the rebuilt artifact layer

The grid was run a second time, end to end, on top of `w98_artifacts`
(`data/g98_g_grid2`): plan validated before any boot, resume verified against
record content, coverage counted from the records. It **self-audited clean**
— 31 cells, 62 records, two replicates each, zero identity mismatches — which
is the first time the coverage claim is checked rather than asserted.

| | grid 1 | grid 2 |
| --- | --- | --- |
| selector / omniscient | 0.9820 | **0.9839** |
| selector / OFF | 1.415x | 1.426x |
| selector | 572.8 tok/s | 578.6 tok/s (+1.01%) |

Per-regime regret reproduces: 1.76→1.84 (R1), 1.30→0.80 (R4), 3.33→1.62 (R5),
0.00→0.00 (R5cot), **10.59→10.99 (R6)**, 0.22→0.00 (R8). The omniscient cell
is identical in five regimes; at R8 it moves between `w512/skip4` and
`woff/skip4`, which is exactly the pair whose measured regret is 0.22% and
0.00% — two cells inside each other's noise, so which one is "best" is a coin
flip and costs nothing.

**Two independent 62-boot grids agree on the headline to 0.2 percentage
points**, and the R6 miss reproduces at both. The result is not an artifact of
one run.

## A bug this run exposed, worth recording

The first pass reported "31 cells" while measuring **30**. `_slug` named
boots by their lever values alone, so the OFF cell and the unlevered armed
cell (`target-matching/woff/skip0`) — identical in every lever, differing
only in whether the draft is armed — collided on one filename. Because a
completed boot is skipped rather than overwritten, whichever ran second was
dropped silently.

Two things made it worse and are the real lesson:

* The collision resolved **differently in each round**, since round 1 runs
  the order reversed: r0 kept the OFF measurement, r1 kept the armed one. A
  blind rename of "the OFF file" was therefore wrong in exactly half the
  cases, and mislabeled an armed measurement as OFF until the records were
  re-read and renamed **by their own `config` field** rather than by name.
* The same bug class was live in the MoE probe at the same time, where
  `_slug` ignored the quant axis and dropped a quantized cell.

Fixed in both: a slug must carry every axis that distinguishes two
configurations, not merely the ones that usually differ. The missing cells
were then measured and the grid re-scored; the headline was unchanged
(0.9820 both before and after), because the unlevered armed cell is not
omniscient in any regime.

## Scope

* **Armed-vs-armed only.** `selector / omniscient` is sound here;
  `selector / OFF` carries roughly ±10% and is reported with that band.
* **One lane.** GPU 1 was occupied for part of the session, so there is no
  cross-lane replicate; the two interleaved rounds are the replication.
* **Same caveats as G98-G**: decode currency, one model, K=4, acceptance
  transferred from h104 on disjoint content seeds.
