# Phase 83 — profiled levers: does optimization flip the map?

Source phases: 79 (menu-extension gate: singles-greedy layer sets already
double skip beta; set-level backtest: profiles nominate, measurements
confirm), 77 (beta harness + refs; RouterMask contiguous-shard semantics),
80/79 (map v4 + winner hardening: every winner is the argmax over NAIVE
lever forms). User review 2026-07-14: the map compares naive instantiations
(contiguous skip, RTN quant, contiguous expert shard); the literature
(SWIFT/KnapSpec; GPTQ/AWQ) shows offline profiling recovers large accept at
equal cost. The map's unit of selection should be lever + PROFILED setting.

## Objective

Measure how much beta offline profiling recovers per lever, and whether
profiled levers FLIP map cells. **Gate: either (a) at least one
OFF/marginal region flips winner with a measured e2e check, or (b) profiled
forms land within epsilon of naive and the map is certified
profiling-robust — both are findings; (a) is the headline.**

Headroom analysis (why these levers): winners (quant/window) sit at beta
0.92-0.99 — bounded upside; losers hold the headroom: skip 0.45-naive ->
0.85 already at singles-greedy; local-route 0.69-0.83 for a CONTIGUOUS
shard [0:n) — routing mass is skewed, frequency-profiled expert selection
should recover substantially. Flip candidate #1: MoE short-ctx OFF region
(current max 1.04-1.07; profiled skip/lr at beta ~0.9, R ~0.6 prices
~1.15-1.25).

## Plan

- **E1 — interaction-aware layer profiling** (dense first, MoE next):
  iterative greedy — re-measure the candidate column after each accepted
  drop (79 measured that interactions break the product at depth; the
  singles ranking is only a round-1 prior). Budgets to 7; candidate pool =
  top-10 unchosen by current-round beta (cost cap, stated). Output: the
  profiled beta-vs-budget frontier vs contiguous + singles-greedy.
- **E2 — frequency-profiled expert selection** (MoE): collect per-layer
  expert-usage counts over ref continuations (hooks, no masking), then mask
  to the TOP-FREQUENCY set per layer (vs contiguous [0:n)) at frac 0.5 /
  0.25. Compare to lr50/lr25 beta.
- **E3 — calibrated quant** (AWQ/GPTQ int4 ckpt): bound the winning
  levers' remaining headroom. (After E1/E2.)
- **E4 — map v5**: profiled columns -> exhaustive search re-run -> count
  flips -> e2e-validate any flip (R side: profiled sets/experts change beta
  ONLY; R carried from the lever's cost family — skip R by depth, lr R by
  fraction — same as naive, so map math is beta-swap only).

## Constraints / traps

- beta method: 77 harness, ondist refs, >=1152 positions, paired vs the
  same refs as the naive arms (deltas are variance-free).
- GPUs shared box: probe before launch; caches on /data.
- Profiled sets must be re-measured, never product-priced (79: depth breaks
  the product; profile-solve mis-prices even when it ranks right).
- Iterative greedy on MoE at 30B is ~3x dense cost — run after dense
  validates the loop.

## Expected next artifact

`results_profiled.md`: E1 frontier + E2 freq-vs-contiguous table, then the
map-v5 flip count and e2e checks.
