# Contribution status vs the paper's four claims — working doc
# (2026-07-23; the agenda for the status discussion)

The four contributions as stated by the user (2026-07-21):

> C1. There is no single lever for self-speculative decoding.
> C2. An EFFICIENT way to find the optimal lever for each regime.
> C3. A system for efficient switching and online adaptation.
> C4. Effectiveness demonstrated at non-RL generation AND RL rollout.

## Scorecard

| # | claim | status | evidence anchor | open gap |
|---|---|---|---|---|
| C1 | no single lever | **CLOSED** | T2/T3 (arch: dense/MoE/MLA, wrong-lever 30-60% loss); T8 (32B kernel split 5-2, depth inversion, triple composition scale-keyed; 9 regimes need K2-K6 + OFF) | hardware axis argued, measured on ONE box (2nd-HW column pending) |
| C2 | efficient search | **SUPPORTED, unconsolidated** | beta harness (1152 pos, no e2e); domination pruning + build-on-selection (w4+ffn never built); 91-min protocol; KnapSpec-recipe backtest; measure-on-deployment (32B transfer failure as positive evidence) | (a) no single cost-to-onboard NUMBER; (b) shallow-K found by EVAL not search -> grid coverage is part of the method, present as audit loop |
| C3 | switching + adaptation system | **CLOSED** | E0 toggle table; compiled policy (regret 1-3%, wins 2/3 traces); 113ms pinned swap, bit-exact graph replay, live refresh 114-116ms, detector-fired | per-step switching = K/OFF + same-layout refresh; cross-kernel = priced boot-class transition (not hot) -- state precisely |
| C4 | non-RL + RL effectiveness | **CLOSED w/ framing discipline** | non-RL: 1.80x/1.90x serving wall, 9/9 regimes, policy trace wins; RL: 1.055x over AR under drift, staleness -45% -> noise, beat-AR recipe + measured anti-patterns | (a) drift EMULATED (calibrated perturbation), not live trainer; (b) demo ran the thinnest cell -- lead with staleness-bound + concentration (+7-33%), not the +5.5% |

## Work items (the task list)

0. **[C2] Phase 90 (hierarchical search)** — the C2 hardening per
   the 2026-07-23 discussion: analytic-cost prefilter -> importance-
   proxy shortlist (rank-validated vs committed betas) -> direct beta
   -> per-regime pools -> switch-cost-aware bandit. First target: R4
   heterogeneous per-layer/head windows (accept headroom 4.06 at
   priced-out cost; free-toggle lever). research/90_hier_search.
1. **[C2] Cost-to-onboard table** — assemble from committed timings:
   beta column cost, e2e arm cost, compile-policy cost per cell,
   solve cost -> "zero to compiled policy on a new column = X
   GPU-hours". No new GPU work; data exists in phase logs
   (86 91-min protocol, 88/89 compile runs ~5-7 min/arm/cell-grid).
2. **[C2] Pre-registration scorecard** — table of every explicit
   prediction vs outcome: 86 P1-P4 (QK-norm rule, skip scale-dep,
   quant-led 8B, 32B composition ~1.5-1.6x), the skip-domination
   prediction (REFUTED by measurement -> composition win), the 32B
   transfer prediction (falsified -> measure-on-deployment), fleet
   +39.1% (prediction, unrun). Predict-fail-correct = the audit loop
   working; honesty asset.
3. **[C1] Hardware-axis honesty pass** — ensure every hardware claim
   is labeled measured-on-H100 / kernel-availability-argued; 2nd-HW
   column stays queued on hardware access.
4. **[C4] RL framing pass** — DIRECTION/T9 wording: emulated drift
   disclosed up front; thinnest-cell choice stated as adversarial
   design; headline = staleness bound + concentration, +5.5% as the
   floor. Live-trainer integration = future work.
5. **[C3] Precision pass** — "what switches when" table (per-step:
   K/OFF; ~100ms: same-layout weight refresh; boot-class: cross-
   kernel/ckpt swap w/ E0 prices) into DIRECTION Sec C.
6. **[paper] Two-paper decision** — the record now spans maps+search
   (C1/C2) and system+adaptation (C3/C4); revisit the recorded
   fallback split vs page budget.

## Discussion questions for the user

- Which of C1-C4 is THE headline (ordering shapes Sec A)?
- Page budget / venue -> one paper or the recorded two-paper split?
- Is the emulated-drift RL demo sufficient for C4, or invest in a
  live-trainer (e.g. open-source RLHF loop) integration before
  submission?
- 2nd-hardware column: wait for access or ship with the one-box
  caveat?
