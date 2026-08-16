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
| D3 re-scored, fail-closed | **99.6%** of omniscient, **1.436x** over static-OFF, **+6.2%** over the best static, on **102 of 126 mixes** | **C3** — the claim "arms correctly and fails closed" now measured rather than asserted; the failing static clause passes |
| surrogate recall | **recall@4 = 12/12**, confirm-1 regret <= **2.0%**; damage budget certifies bound tightness (rho **+0.875**) but NOT ranking | **C2** — the search's ordering carries a stated guarantee at a stated shortlist size, which is what answers "yours is a heuristic, KnapSpec's is a solver" |

**Three results the paper should absorb.**

1. **The value decomposition, and an exact law for the switching term.**
   Best single lever 1.20x over OFF; best three-lever composition 1.35x;
   per-regime selection **1.44x** once the selector can decline. So
   **composing is worth +12.5% and switching +6.2%.** (An earlier reading of
   this record put switching at +1.4%; that was net of a 24% self-inflicted
   regression at R4 and is withdrawn.) C3's claim should still be scoped to
   "arms correctly and fails closed" — that is now the *measured* headline,
   not a fallback.

   The switching term obeys an exact identity under time-weighted
   aggregation: `gain = sum_R t_R * (rate_sel(R)/rate_static(R))` with `t_R`
   the **time** share. Switching therefore pays only where slow regimes have
   distinctive optima, and this grid never has both at once — R1 carries 59%
   of the time and contributes +0.14%. The fail-closed selector already
   captures 93% of all switching value the grid contains, so the ceiling is
   a property of the measured regimes, not of the design.
2. **C1 visible inside ONE model.** The window lever is worth 1.01-1.09x at
   short context and **1.45-1.52x at 14k**, while quantization is
   universally on. "No universal lever" does not need the cross-architecture
   grid to show itself.
3. **A located failure of fail-closed, and its repair.** At R4 every lever
   family loses and the selector armed anyway at **0.76x** — its predicted
   margin was the thinnest in the map (1.03x), so the signal was present and
   unread. Arming only when the margin's lower confidence bound clears unity
   — both error terms estimated from pre-D3 data — fires at exactly that one
   regime and takes the selector to **99.6% of omniscient**. C3 gets to
   claim a fail-closed selector because one was built and scored, and the
   paper gets the predict/fail/correct loop on its most damaging miss.

4. **The search's guarantee is recall, not top-1.** No rule ranks the argmax
   first reliably — 75-100% top-1 miss for every rule including ours, this
   record's own result. The defensible claim is that the *shortlist contains
   the optimum*: **recall@4 = 12/12** with confirm-1 regret bounded at a
   measured 2.0%. A registered attempt to do better — a damage budget
   gating where the surrogate is valid — was **refuted**, and what survived
   is that the budget certifies the product bound's *tightness*
   (rho +0.875, bound always in the sound direction). This is the honest
   answer to "KnapSpec has a solver and you have a hunch".

**Caveats to carry.** One model, one box; decode currency only (prefill is
46.6% of wall at R5); equal-weight mix is a convention, with the
mix-frontier reported alongside; the fail-closed rule is post-hoc with a
pre-D3-data-only derivation (two invariance checks stand in for a barrier);
the 1.305x mixed-feasibility figure is modelled over measured rates, not
measured; recall@4 rests on 12 groups (95% lower bound 0.78).

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
