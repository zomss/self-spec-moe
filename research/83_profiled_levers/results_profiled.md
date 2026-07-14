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

## E1-moe — the layer profile is FLAT; profiling structure is architecture-specific

Leave-one-out column (44 layers): 0.946-0.959, no early/late structure
(dense: 0.836-0.954 with early-droppable/late-critical shape). Iterative
greedy frontier: 0.959 / 0.935 / 0.917 / 0.898 / 0.869 / 0.742 / 0.683
(budgets 1-7; interaction cliff between 5 and 6). At the skip125-equivalent
budget (6/48), profiled = 0.742 vs naive contiguous ~0.70: **+0.04, vs
dense's +0.42** at equal relative budget. No MoE cell flips via layer
profiling (best pricing ~0.95x).

**The architecture-split law**: profiling headroom lives where the
parameters are — dense profits from layer PLACEMENT (redundancy
concentrated in early blocks), MoE from EXPERT selection (redundancy
concentrated in routing skew; layer redundancy spread flat by design —
each layer's contribution is already diluted across 128 experts). Even the
STRUCTURE of profitable profiling fails to port across architectures —
the portability verdict (77-F5) extends from lever values to profiling
strategies.

## E2b — e2e of the priced flips: ACCEPT PROVEN, delivery blocked by the realization

Plumbing (committed): VLLM_SELF_SPEC_DRAFT_RESIDENT_SETS masks draft
routing to per-layer frequency sets (requires the draft full replica).
Finding #0: **the bf16 full replica does not fit** on 4x80GB beside the
EP4 target (weights 73.25 GiB/rank) -- the runnable realization is the
fp8 replica (= the flr50+q_fp8 combo, itself priced 1.11x).

b4/2k and b32/2k, 8-iter pairs (gamma=2, mitigation stack on):

| cell | nospec | flr50q+mitig | ratio | accept (pred ~2.82) |
|---|---|---|---|---|
| b4/2k | 460 +-12 | 427 +-84 | 0.93x | 2.851 |
| b32/2k | 2487 +-33 | 1859 +-18 | 0.75x | 2.793 |
| (naive lr, b4/2k, 4-iter) | 475 | 259 | 0.55x | 2.173 |

1. **The acceptance flip is fully proven e2e**: 2.17 -> 2.85, exactly the
   offline surface's prediction (beta 0.953 x 0.99), stable across four
   runs and both cells. Offline-profiled routing transfers to the real
   system with no loss.
2. **Delivery fails for attributable, lever-external reasons**: at b4 the
   windowless-MoE chain floor leaves the cycle ~2 ms short of break-even
   (0.93x); at b32 the fp8 replica's small-M kappa (the map's own F2/V3
   lesson) lifts the realized draft R to ~1.0 vs the priced 0.83x0.95 --
   the REALIZATION, not the profiling, broke the pricing.
3. **The correct realization is a bf16 PARTIAL replica** -- load only the
   top-frequency experts' weights (frac 0.5 = ~28 GiB bf16, fits): bf16
   GEMMs (no kappa), comm-free, and less memory than the fp8 full replica.
   Why replicate experts the mask never routes to? Future plumbing (weight
   loader surgery), now with a measured motivation.

**Gate verdict**: condition (a) NOT closed -- flips are acceptance-proven
and pricing-correct on the beta side, but delivered throughput needs the
partial-replica loader (R side) and/or the MoE-chain floor fix. Both gaps
are quantified and lever-external. The map-v5 entries carry this
provenance: "flr50: beta measured e2e; R pending realization".
