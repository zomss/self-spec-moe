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

## E2c — bf16 PARTIAL replica: the loader works; ONE delivered flip; the band's boundary measured

Plumbing (committed 51918f8b2 + draft_model override): per-layer expert
maps installed at FusedMoE construction inside the proposer's build
context -> parameter tensors sized to the kept set, loader's -1 skip
drops non-resident experts. **Verified end to end: model load 73.25 ->
46.25 GiB (exactly half the expert bytes), draft layers report
use_ep=False + expert_map set, accept 2.88 (the highest of any
realization -- bf16 numerics, no fp8 noise).**

The realization ladder at the flipped band (2k ctx, gamma=2, vs nospec):

| cell | priced | naive lr | fp8 full replica | bf16 partial replica |
|---|---|---|---|---|
| b4/2k | 1.11x | 0.55x (2.17) | 0.93x (2.85) | **1.03x (2.882) — DELIVERED** |
| b8/2k | 1.11x | — | — | 0.63x (2.841) |
| b32/2k | 1.09x | — | 0.75x (2.79) | 0.64x (2.811) |

1. **First delivered flip**: b4/2k at 1.03x (+-0.10 -- parity-to-win),
   +11% over the fp8 realization at identical routing = the kappa
   attribution CONFIRMED at low batch.
2. **kappa is not the whole story at scale**: at b32 the bf16 partial is
   WORSE than fp8 full (0.64 vs 0.75) -- with ~all 64 residents activated
   per step, byte reads match the fp8 replica without its byte savings.
   At b8+ the draft chain's cost grows faster than verify's: the needed
   cycle is 27.4 ms, measured 43.6 ms.
3. **The mechanism, named**: the map's lr R (0.83) is a STANDALONE-serve
   number; the W7 draft CHAIN at short ctx is far above it (the
   delivery(gamma, R) law, MoE edition: at 2k the verify is too cheap to
   amortize any chain overhead at batch >= 8). Full band delivery needs
   MoE chain-execution work (the 81-class floor/step-cost program,
   windowless variant) -- lever-external, quantified per cell.

**Phase-83 gate, final**: condition (a) MET at b4/2k (measured 1.03x
where the naive map said OFF at 1.04x priced/0.55x realized), with the
acceptance flip proven at every cell (2.81-2.88 across the band, exactly
the offline surface). The stronger statement stands regardless of
delivery: profiled levers change WHAT THE MAP SAYS (4 cells), the beta
transfer is exact, and the remaining gaps are execution, not selection.

## E3 — calibrated quant: GPTQ buys +0.019 beta; winners unchanged

GPTQ W4A16 (int4 sym g128, 512 C4 samples; `make_gptq_ckpt.py` in the
CUDA lc venv) scored on 77's dense refs, decompressed forward:

| variant | beta (dense, 16k) |
|---|---|
| RTN fake-quant (77, = the deployed data-free RTN ckpt) | 0.924 |
| **GPTQ-calibrated** | **0.9427** |

+0.019: real but modest — RTN was already near the lever's ceiling on
this model. Map effect: dense single-quant cells +0.02-0.04x (1.28 ->
1.31-1.32); composed cells' measured-combo beta pins the headline cells
unchanged. Verdict: calibration is worth taking when the ckpt is built
anyway; it flips nothing. (Note for the record: the deployed W4 artifact
is data-free RTN, so map beta and artifact were consistent all along.)

## E4 — strategy map v5 (data/strategy_map_v5.md): 4 flips, all profiled-expert-driven

Exhaustive search with profiled columns (dense ls3 0.869 + GPTQ q_int4
0.9427; moe flr50/flr25/skip6p; mla unchanged):

- **4 flips vs v4, all on MoE, all flr50-driven**: b4/2k, b8/2k, b32/2k
  -> flr50+q_fp8 (1.12-1.15x priced; e2e: b4 DELIVERED 1.03x, b8/b32
  chain-blocked) and b4/16k -> flr50+q_fp8+win512 (1.11x, priced).
- Dense/MLA winners unchanged (values +0.02-0.04 from calibration) —
  consistent with the architecture-split law: the profiled lever that
  changes decisions is the one aligned with where the architecture's
  redundancy lives.
