# Results: Rebalancing Ceiling vs Replication Budget

Date: 2026-06-24

## Status

Acceptance vs per-device replicated draft-cache budget `M`. The cache is the
per-layer top-`M` experts by gate mass, replicated on every device (EP-invariant
by construction). This is the **mass-optimal fixed placement**, so it upper-bounds
any rebalancing/replication scheme. One-token proxy (optimistic).

## Qwen3-30B-A3B (E=128, top-8)

| M | M/E | gamma | top-k overlap | sampled acc | top-1 (greedy) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 0.06 | 0.227 | - | 0.147 | 0.125 |
| 16 | 0.12 | 0.329 | - | 0.226 | 0.125 |
| 24 | 0.19 | 0.415 | 0.515 | 0.363 | 0.188 |
| 32 | 0.25 | 0.491 | 0.607 | 0.470 | 0.375 |
| 48 | 0.38 | 0.623 | 0.739 | 0.693 | 0.688 |
| 64 | 0.50 | 0.734 | 0.836 | 0.781 | 0.688 |
| 96 | 0.75 | 0.905 | 0.952 | 0.911 | 0.875 |
| 128 | 1.00 | 1.000 | 1.000 | 0.998 | 0.938 |

## GPT-OSS-20B (E=32, top-4)

| M | M/E | gamma | top-k overlap | sampled acc | top-1 (greedy) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 0.12 | 0.209 | 0.277 | 0.362 | 0.312 |
| 8 | 0.25 | 0.353 | 0.439 | 0.537 | 0.500 |
| 12 | 0.38 | 0.487 | 0.564 | 0.558 | 0.438 |
| 16 | 0.50 | 0.611 | 0.663 | 0.625 | 0.438 |
| 24 | 0.75 | 0.830 | 0.859 | 0.758 | 0.750 |
| 32 | 1.00 | 1.000 | 1.000 | 1.000 | 1.000 |

Sanity: `M = E` gives acceptance ~1.0 (no masking), and the curve is monotonic.

## Key numbers

**Acceptance at the plain-EP budget `M = E/N`** (what each device holds for free):

| | EP8 (M=E/8) | EP4 (M=E/4) | EP2 (M=E/2) |
| --- | ---: | ---: | ---: |
| Qwen3-30B | 0.226 | 0.470 | 0.781 |
| GPT-OSS-20B | 0.362 | 0.537 | 0.625 |

**Budget needed to reach `beta >= 0.8`** (linear interpolation):

| Model | `M*` | `M*/E` |
| --- | ---: | ---: |
| Qwen3-30B | ~69 | ~0.54 |
| GPT-OSS-20B | ~25 | ~0.79 |

So reaching useful acceptance requires replicating **roughly half to four-fifths
of all experts on every device.**

## The replication factor scales with EP

The cache is EP-invariant, so `M*` is fixed (~0.5-0.8 E) regardless of EP. The
required **replication factor** relative to the plain-EP per-device budget is:

```text
R = M* / (E/N) = N * (M*/E)  ~=  0.5 N  ..  0.8 N
```

`R` grows **linearly with EP size N**:

| EP size N | R (Qwen3, M*/E=0.54) | R (GPT-OSS, M*/E=0.79) |
| ---: | ---: | ---: |
| 2 | 1.1x | 1.6x |
| 4 | 2.2x | 3.2x |
| 8 | 4.3x | 6.3x |
| 16 | 8.6x | 12.6x |
| 32 | 17x | 25x |

At the cross-node EP scale (N >= 8-32) where the communication advantage is
large, reaching `beta >= 0.8` needs `R = 4x..25x` the per-device expert memory
that EP exists to save — i.e. essentially full replication on every device.

## Interpretation

- **Your hypothesis was directionally correct:** acceptance rises cleanly and
  monotonically with local expert coverage. More local experts -> higher
  acceptance. The mechanism is exactly per-device capacity.
- **But the quantity needed is prohibitive:** it is not a modest rebalancing.
  Useful acceptance requires ~half the experts local, and the replication factor
  to hold that at high EP grows as ~N/2. That negates the memory rationale for EP
  and shrinks the all-to-all being amortized (if most experts are local, the
  baseline verify is cheaper too).
- **The trade-off is worst exactly where the method should win:** higher EP gives
  more exposed communication (more upside) but demands a larger replication factor
  (more cost) for the same acceptance. Cost and benefit both scale with N.
- This is the mass-optimal fixed cache and a one-token proxy, so the real picture
  (heuristic placement, multi-token, greedy decode where top-1 is even lower) is
  strictly worse.

## Conclusion

The rebalancing ceiling confirms and quantifies the Phase 07 negative result.
Affordable rebalancing (small replication over the EP budget) lands acceptance
around 0.3-0.5; reaching `beta >= 0.8` requires near-full per-device replication,
which is infeasible precisely in the high-EP regime the method targets. The one
mechanism that escapes the `R ~ N/2` scaling is an **EP-invariant shared expert**
(fixed local mass independent of replication), which remains the only untested
lever.
