# Phase 83 results

## E2 — frequency-profiled expert selection: +0.13 beta, RESIDENT SET HALVED

Usage skew measured (4 ref prompts, all 48 routers): top-50% experts carry
96.3% median routing mass (min 0.835). Profiled keep-sets vs the naive
contiguous shard [0:n), same refs (paired):

| arm | contiguous (77) | frequency-profiled | delta |
|---|---|---|---|
| 50% resident | 0.826 | **0.9531** | +0.127 |
| 25% resident | 0.693 | **0.8229** | +0.130 |

- **flr25 ≈ lr50**: a quarter of the experts, frequency-chosen, matches the
  naive half-shard — resident memory/dispatch halves at equal accept.
- flr50 (0.953) approaches window (0.971): the accept-preserving comm-free
  lever the P24-29 line wanted, recovered by profiling.
- Map v5 pricing (beta swap; R = measured m_localroute per cell, identical
  cost structure): **the MoE short-ctx OFF band FLIPS** — b4/2k 1.11x
  (was 1.04), b8/2k 1.11x (1.06), b32/2k 1.09x (1.07), b4/16k 1.10x
  (1.05); window unchallenged at >=16k high-batch. Flips are PRICED —
  e2e check requires harness plumbing for frequency-selected resident
  sets (next work item).
- Comm-bound fabric implication (deferred tier): the comm-free triple's
  beta ceiling moves 0.82 -> ~0.92 (product with win+fp8).

## E1 — interaction-aware iterative greedy (dense): placement, not contiguity

Frontier (re-measured column each round, POOL=8):

| budget | set | beta | singles-greedy | contiguous |
|---|---|---|---|---|
| 1 | {5} | 0.954 | — | — |
| 3 | {4,5,6} | 0.869 | 0.849 | 0.448 (middle) |
| 5 | {3,4,5,6,12} | 0.761 | — | — |
| 7 | {3,4,5,6,7,11,12} | 0.557 | 0.507 | 0.09 |

The chosen set is a nearly CONTIGUOUS EARLY block ({3..7}+{11,12}) — the
79-era reading "set selection beats contiguous" sharpens to **placement is
what the naive arm got wrong**: the SWIFT-style middle-block convention is
the mistake; early layers drop in blocks almost freely. Iterative greedy
adds +0.02/+0.05 over singles-greedy at budgets 3/7 (interaction-aware
re-measurement pays modestly on dense). Dense skip remains map-irrelevant
(winners are 0.92+ quant/window); the profiled frontier's value is on MoE
(skip runs +0.24 higher there) — E1-moe running.
