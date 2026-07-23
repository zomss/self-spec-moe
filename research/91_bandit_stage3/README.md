# Phase 91 — RL-performance push + stage-3 bandit formalization

Source: user directive 2026-07-24. Two coupled goals:
(a) push the RL drift-trace result past 1.055x (89-E3b v4), and
(b) replace the hand-tuned scheduler argmax (EMA + asymmetric
hysteresis + probe bursts) with a principled switch-cost-aware
non-stationary bandit (literature: BanditSpec / Not-a-Bandit for
external-drafter pools; ours = self-spec config pools + measured
toggle costs + DRAM refresh).

## Identified RL headroom (measured basis)

1. K-GRID: the b16 ctx progression in the compiled Hum table flips
   winners along a generation (ctx2k: K3 1.06 / K4 0.94-loss;
   ctx8k: K4 1.49; ctx14k: K4 1.92). A 2k-prompt 3k-gen spends most
   TOKENS in K4-favorable ctx; deployed kmax=3 cannot reach it.
   E5b: kmax=4 table arm (width-5 padding tax at the start vs K4
   gains later — measure the net).
2. DETECTION LATENCY: client-side 10s poll leaves ~1% aggregate in
   the deep-drift dips. The bandit's posterior IS the in-engine
   drift signal; scheduler-side disarm is instant already — the
   refresh trigger stays client-side (cheap) but keyed off faster
   signals.
3. TRACE REALISM: RL rollouts drain (variable lengths, T=1.0).
   Deeper drain visits spec-favorable small-batch cells. Verify the
   drain profile of the E3b trace; report sensitivity.

## Bandit design (E5)

Per (cell, arm): discounted Beta posterior over per-position accept f
(the accept-EMA generalized: EMA = posterior mean; half-life = the
discount). Per step: Thompson-sample f_a per arm, compute S_a =
(1+f_a*K)/(K*R_a+1) from compiled R, switch-cost-aware argmax
(arming pays the measured 2%; disarm free). OFF-arm exploration:
discounting widens the OFF-state posterior -> Thompson naturally
re-probes; replaces hand-tuned probe bursts (8/128) with calibrated
exploration.

## Plan

- E5a (CPU): calibrated simulator — per-cell accept streams matched
  to measured moments (compiled f_ref + drift-trace phase means);
  compare argmax-EMA (current), Thompson, sliding-window UCB, each
  +/- switch-cost awareness. Deliverable: regret table; gate =
  Thompson >= argmax-EMA everywhere and beats it under drift.
- E5b (GPU): kmax=4 drift-trace arm (RL headroom item 1).
- E5c (engine): VLLM_SELF_SPEC_BANDIT=1 scheduler option (Thompson
  in place of argmax; same tables, same EMA state reinterpreted as
  posterior). Small diff by design.
- E5d (GPU): live A/B on the E3b drift trace: argmax-policy vs
  bandit, both + refresh; and the combined best (bandit + kmax4).
  Gate: bandit >= policy - 1% on every phase (no-thrash) and beats
  it on the trace aggregate.

## Constraints

GPUs 0,1,6,7; caches /data; e3_rl_demo machinery reused; commit
[W7][91]; DRAFT flags.
