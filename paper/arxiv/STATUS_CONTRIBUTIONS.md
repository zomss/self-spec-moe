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
| C2 | efficient search | **CLOSED (2026-07-24, phase 90)** | the measurement theorem (five-way proxy falsification + non-additivity control + on-policy/collectivity mechanisms); T10 cost-to-onboard (~2-4 GPU-h/column); T11 scorecard (7 confirmed/5 refuted->corrected); KnapSpec surrogate-class explanation of the h2h margin; literature convergence corroboration | E1f transfer-prior (0.43) optional footnote; bandit formalization (stage 3) still queued |
| C3 | switching + adaptation system | **CLOSED + FORMALIZED (phase 91)** | E0 toggle table; compiled policy (regret 1-3%, wins 2/3 traces); 113ms pinned swap + live refresh; T12 controller ladder: Thompson bandit built/sim-validated/live-ablated -> deployed argmax necessity-proven as measured optimum; exploration-for-detection finding | per-step = K/OFF + same-layout refresh; cross-kernel = boot-class; kmax-4 grid on Marlin in flight |
| C4 | non-RL + RL effectiveness | **CLOSED w/ framing discipline** | non-RL: 1.80x/1.90x serving wall, 9/9 regimes, policy trace wins; RL: 1.055x over AR under drift, staleness -45% -> noise, beat-AR recipe + measured anti-patterns | (a) drift EMULATED (calibrated perturbation), not live trainer; (b) demo ran the thinnest cell -- lead with staleness-bound + concentration (+7-33%), not the +5.5% |

## Phase 98 — the two-round selector, independently scored (2026-08-16)

A self-contained demonstration of the two-round design on Qwen3-8B dense,
preregistered end to end with commitment barriers, on bare metal. NOT a
re-run of C2/C3's data: fresh lattice, fresh prompt freeze, disjoint content
seeds, decode currency, equal work. Per-claim records in
`research/98_selector_demo/`.

| claim | result | evidence for |
|---|---|---|
| D1' cost soundness | 47/48 held-out covered (97.9%), median error 0.53%; the sound elimination rule **exercised 5x, correct 5/5** | **C2/C3** — the cost half of the epistemic split, with the rule finally EXERCISED (Round 1 could only report it untested) |
| D2(a) screen honesty | 6 violations / 132 rows, **all carrying window=128 at short context**; interaction ratio median **above 1 for every lever set**, 1.670 at R4 | **C2** — corroborates P2's corrected direction: composed acceptance EXCEEDS the product, so the bound admits but cannot eliminate |
| D2(b) knapsack identity | 11/12 controls beaten; product-of-singles ranks **perfectly (rho +1.000) at k=4 and k=8**, then **inverts at k=16** | **C2** — the measurement theorem reproduced on a new lever, model and box, with the non-additivity boundary now LOCATED (between 8 and 16 of 36 layers) |
| D3 end-to-end | **95.1%** of the omniscient composite, **1.371x** over static-OFF; beats every static by +2% in only **9 of 126 mixes** | **C3** — a clean replacement for the retracted "two-round recovers a win the oracle missed" exhibit, and a price on switching |

**Three results the paper should absorb.**

1. **The value decomposition.** Best single lever 1.20x over OFF; best
   three-lever composition 1.35x; per-regime selection 1.37x. So
   **composing is worth +12.5% and switching +1.4%.** C2 is where the value
   is, and C3's claim should be scoped to "arms correctly and fails closed",
   not "switching beats statics".
2. **C1 visible inside ONE model.** The window lever is worth 1.01-1.09x at
   short context and **1.45-1.52x at 14k**, while quantization is
   universally on. "No universal lever" does not need the cross-architecture
   grid to show itself.
3. **A located failure of fail-closed.** At R4 every lever family loses and
   the selector armed anyway at **0.76x**. C3 claims a selector that fails
   closed; this is a measured instance where it did not, with the diagnosis
   (its predicted margin was the thinnest in the map, 1.03x) and the fix
   (a margin-keyed refusal) both identified.

**Caveats to carry.** One model, one box; decode currency only (prefill is
46.6% of wall at R5); equal-weight mix is a convention, with the
mix-frontier reported alongside; D3's static clause depends on a reading of
its own preregistration and awaits a ruling.

## Work items (the task list)

0. ~~[C2] Phase 90 (hierarchical search)~~ DONE 2026-07-24: proxy
   program closed (five-way falsification -> the measurement
   theorem); hetero-window refuted by its own gate (collectivity);
   E4 canceled at zero cost; R4 headroom -> kvq draft-ctx.
1. ~~[C2] Cost-to-onboard table~~ DONE -> T10:
   beta column cost, e2e arm cost, compile-policy cost per cell,
   solve cost -> "zero to compiled policy on a new column = X
   GPU-hours". No new GPU work; data exists in phase logs
   (86 91-min protocol, 88/89 compile runs ~5-7 min/arm/cell-grid).
2. ~~[C2] Pre-registration scorecard~~ DONE -> T11 (14 entries):
   originally: table of every explicit
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
