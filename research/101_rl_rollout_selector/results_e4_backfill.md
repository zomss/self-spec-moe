# E4 — the registered request set removes the draw variance, and invalidates the lattice

Date: 2026-08-23. Eight boots: `w1024/skip4` and `w1024/skip8` on the same
four disjoint LO draws as E3, now with **n = 32 submitted against 8
concurrent** — the Phase-100 registered `n = 4 x max batch`.

## The defect

Phase 100 registers: *"Fixed request set per cell, n = 4 x max batch (long
cells n=64, SS n=256), submitted at once; continuous batching drains the
set."* This runner submitted exactly `BATCH` requests and set
`max_num_seqs >= 32`, so concurrency exceeded the request count and the batch
could only drain, 8 -> 1, with nothing to backfill. **Every measurement in
phases 98-101 ran that way.**

## The result

| quantity | E3 (n=8) | **E4 (n=32)** |
| --- | --- | --- |
| occupancy sd | 23.3 | **2.1** |
| throughput sd | 22.8 | **3.6** |
| tau sd | 2.3 | 0.7 |
| mean occupancy | ~4.2 of 8 | **6.0-6.6 of 8** |

Per-draw throughput deltas (skip8 vs skip4): **+0.8%, -0.0%, -3.5%, -7.0%**
against E3's -4.7%, +21.8%, +31.3%, -17.9%. **The 49-point band becomes 7.8
points, and every draw now agrees on the sign.**

Mean: skip4 ahead by **2.4%**, consistent with tau (-10.2%, sd 0.7) partly
offset by skip8's ~8% cheaper draft chain. That is the equal-work answer,
recovered under the deployment protocol.

**So `w1024/skip8` never was the best arm at LO.** It won section 45 because
that grid drained an unbackfilled batch of eight on one prompt draw, a
configuration with +-23% occupancy noise. The seven-hypothesis chain closes
here.

## What this invalidates

| claim | status |
| --- | --- |
| section 45's per-point winners, "seven distinct winners" | **needs re-measurement** |
| the +7.28% equal-mix ceiling and +19.8% mix gaps | **needs re-derivation** |
| section 48's shortlist scoring and confirmation-budget curve | rests on the above |
| E1's trajectory value, E1c/E1d bands | rest on the above |
| cost fits, tau curves, section 49's K sweep, section 25's equal-work comparison | **unaffected** |

Absolute levels move too: per-token cost at LO drops from ~1.60 to ~1.25 ms
because occupancy sits at 6.3 rather than 4.2. Numbers from the two protocols
are not comparable.

## Why this was invisible for so long

Three checks that should have caught it did not, and each was answering a
different question:

* **Boot replication (s44)** reuses the same prompts, so it measures
  boot-to-boot noise (1.31%) and is blind to draw variance.
* **Equal work (s25, E1d, E1e)** fixes occupancy by construction, so it
  measures the arm correctly and cannot reveal that the natural-EOS protocol
  is noisy.
* **The divergence gate** checks token identity, not drain shape.

The disagreement between equal-work and natural-EOS rankings was visible from
section 25 onward and was read as a workload-contamination effect. It was
that, plus an unregistered request count nobody had compared against the
protocol document.

## Scope

Two arms, one cell, one batch, four draws. The variance estimates rest on
four points each; the collapse from 23.3 to 2.1 is far outside that
uncertainty, and the sign agreement needs only two draws.
