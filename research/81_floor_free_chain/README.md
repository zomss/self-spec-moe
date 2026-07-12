# Phase 81 — cash the composed frontier: a floor-free draft chain

Source phases: 80-E3 (the motivating miss: composed dense draft measured
1.48× vs 1.91× roofline at b32/16k — delivery(γ,R): the PIECEWISE chain's
per-draft-step launch floor dominates a byte-cheap draft; **~1.9× measured
payoff is parked behind this floor**), 74/72 (the floor anatomy + the old
"CG is not the lever" verdict — which was MoE-SPECIFIC), 35/69/70 (the FA3
freeze blocker + the two attempted fixes).

## Objective

Remove (or largely amortize) the per-draft-step fixed cost of the chain so
the composed dense draft (W4-Marlin + window-KV) delivers its measured
byte-cost advantage e2e. **Gate: dense b32/16k composed config measured
≥1.75×** (currently 1.48×; single-window 1.54×; roofline 1.91×). Secondary:
recalibrate delivery(γ, R) on the fixed chain (feeds roadmap item b).

## Why P74's "CG is not the lever" does NOT apply here

P74 measured the MoE draft: its ~11 ms floor was EXECUTION-bound (~2400 tiny
per-layer expert kernels, 86% GPU-active) — CUDA graphs remove launch idle,
not kernel count. The COMPOSED DENSE draft is the opposite profile: 28
layers, no expert loop, W4 weights (5.5 GB) + 528-token window KV → the
GPU work per draft step is ~1-2 ms while the chain step costs several ms —
mostly launch/orchestration. This is the profile CUDA graphs were made for.
E0 verifies this premise BEFORE building anything.

## Plan

- **E0 — floor anatomy (measure first)**: kineto trace of the composed draft
  chain in the harness (P74 tracing tooling, profiler-tax caveats respected:
  trace runs are for decomposition only, never quoted as tok/s). Deliverable:
  per-draft-step breakdown {GPU kernels, launch gaps, python/metadata rebuild,
  sampler} at b8/b32. Gate to proceed: launch+orchestration ≥ 50% of the
  chain step.
- **E1 — the chain-CG A/B, cheapest route first** (P74 ranked these):
  1. **TRITON_ATTN draft chain** — decode grid is seq-len-independent →
     CG-replay-safe where FA3's frozen launch geometry is not (the P35
     accept-collapse trap). Risk: frozen split-K at small batch. The A/B is
     accept_len(TRITON chain-CG) vs PIECEWISE — accept must be UNCHANGED
     (the P35 signature of failure is accept 4.9→1.9).
  2. P69's masked-SDPA fixed-shape scratchpad — measured SLOWER on MoE at
     16k (P70) but never tried on dense with a 528-token window (the
     scratchpad size IS the window — tiny). Re-evaluate.
  3. Bucketed multi-graph chain (one graph per chain position, K≤6) — most
     engineering, only if 1-2 fail.
- **E2 — e2e re-run of the 80-E3 cells** on the fixed chain: b32/16k K=4/6,
  b8/16k K=4 (+ the MoE window cells as a no-regression check). Gate above.
- **E3 — delivery(γ, R) recalibration**: refit the delivery model on the
  new chain from all e2e points; update the search objective (this is the
  bridge to roadmap item b — the selector becomes delivery-aware).

## Constraints / traps (from the record)

- P35: FA3 chain-CG silently collapses accept (frozen launch geometry
  attends truncated context) — every E1 variant ships with an accept A/B
  gate, never step-time alone.
- Profiler and batch-invariant taxes (P74's bogus-headline lesson): clean
  runs for numbers, traced runs for anatomy.
- Box: GPUs 0/1 only while 6/7 are occupied; caches on /data (memory).

## Roadmap context (user-set, 2026-07-14)

(a) THIS PHASE — floor-free chain, cash the ~1.9×.
(b) Effective+efficient search: delivery-aware objective (E3 here), then
    smarter search only if the config space grows (per-layer assembly, 80-E4).
(c) A real switching system: runtime lever selection per request/regime —
    serving integration where the map/selector becomes a scheduler policy
    (lever switch = draft reconfig; needs (a)'s chain + cheap lever toggling).
(d) RL post-training adoption: the rollout regime IS the map's favorable
    corner (long generation, shrinking batch — P74's RL thesis); integrate
    the selector into an RL framework's rollout engine (temperature-aware β
    from 77's T=1 overlap column).

## Expected next artifact

`results_floor.md`: E0 anatomy table, the E1 A/B verdict (accept + step
time per variant), E2 e2e numbers vs the 1.75× gate, the recalibrated
delivery model. Then the paper's §7 headline becomes a measured ~1.8-1.9×.
