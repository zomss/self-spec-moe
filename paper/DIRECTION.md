# Paper direction (single paper) — decided 2026-07-18

Owner's mental model (recorded verbatim in spirit, refined against data):

1. **Self-spec has diverse strategies** (sparse attention, quantization,
   pruning, expert restriction, layer sets, model-free drafting), each
   with setting-level detail.
   - DATA: confirmed; pool complete BY TAXONOMY (84): every
     (cost-term x reduction-op) cell is in-map / gated-dead / excluded /
     scoped, incl. gated-dead levers (pruning, SVD) and a retracted one
     (vres — circular gate, caught by the audit).

2. **The optimal strategy differs on hardware, architecture, workload
   (batch, context — and, per data, output-content distribution).**
   - DATA: arch/batch/ctx axes fully measured (crossovers, portability
     verdict, architecture-split law). HARDWARE: unmeasured (one 4xH100
     box) — REQUIRED addition: >=1 second-GPU R column via the 91-min
     protocol. CONTENT: measured to matter (ngram wins on repetitive
     output; vres beta is coverage-conditional) — the regime signal
     gains an output-distribution feature.

3. **The paper = search strategy + system integration (fast switching).**

   3.1 Search: profile importance (correlated with accept length) and
   cost; find the best setting per regime — the layer-pruning recipe
   (KnapSpec is this, within one lever).
   - DATA-CORRECTED RECIPE: profile-then-SOLVE is measurably
     insufficient — (a) the importance prior decays with depth
     (set backtest: Spearman .94->.60, 30% value error at the argmax);
     (b) cost is REALIZATION-intrinsic, not strategy-intrinsic (same
     lever: 0.55x/0.93x/1.03x across three realizations; kappa(M), chain
     phi/psi, residency); (c) the profiler's own artifacts can be wrong
     (disjoint-artifact rule). OUR RECIPE: nominate-by-profile,
     price-realizations-under-uncertainty (LCB), confirm-by-measurement
     (1-2 per decision), audit loop closes the map.

   3.2 System integration: switching must be prompt; switching costs are
   real and feed the search objective — model (re)upload for quantized
   drafts, CPU-side KV requantization + upload for kv-quant switches,
   both-resident memory rent vs swap latency.
   - DATA IN HAND: load times (46.25 GiB partial replica), residency
     model (7/7 incidents), dwell-time law sketched (82 README:
     dwell* = toggle_cost / delta_rate). MEASURED 2026-07-18 (82-E0,
     research/82_runtime_switching/results_e0.md): OFF and K = FREE
     (per-step scheduler primitives, incl. the fork's per-batch-size K
     schedule); window = hot-switchable (~2 ms, accept-exact restore);
     draft swap = 0.11 s pinned-staged (5.65 GiB @ 50 GiB/s) vs
     both-resident rent; CPU KV-requant (C9) = ~8 s at b8/16k ->
     kvq is deploy-time, NOT a runtime toggle. A switch is a
     REALIZATION change -> the execution-extended pricing is the
     switching cost model.

4. **Eval: normal inference + RL rollout.**
   - THE UNDERWEIGHTED ARGUMENT (promote to motivation): trained-draft
     methods go STALE as the policy updates; training-free self-spec
     levers track the policy for free. RL rollout also exercises every
     axis at once: batch drain, context growth, and content drift over
     training (early repetitive -> ngram-favorable; later not) — the
     switching system has real work to do within one training run.
   - DATA ANNOTATION 2026-07-19: the drift model (predict_switch_gap)
     says monotone within-run drift is tracked by ONE static (-2% for
     switching); RL's argument for the paper is trained-draft
     STALENESS (the lever family tracks the policy for free), not
     runtime switching. Kept as motivation with that framing.

## Single-paper structure (foundation = research/79_paper/paper_draft.md)

- Sec A (foundation, compressed from current secs 3-6): the two surfaces,
  the taxonomy-complete pool, the realization-priced model, the LCB
  selector + audit loop. The current 9.4k-word draft compresses to
  ~40% here; maps become the INPUT to the system, not the headline.
- Sec B (search): the corrected recipe vs profile-then-solve, with the
  KnapSpec head-to-head (our backtest's m=0 row IS their recipe) and the
  91-min protocol.
- Sec C (switching system) — REFRAMED 2026-07-19 after the gap
  prediction (82/scripts/predict_switch_gap.py, measured cells only):
  * The EFFECTIVENESS headline is CROSS-COLUMN selection: one global
    lever config across a dense+MoE+MLA fleet leaves +39.1% on the
    table (measured cells; wrong-lever costs 30-60%: MLA self-spec
    0.56x, MoE dense-chain 0.63x). The map IS the selector; this is
    Sec A/B's payoff restated as deployment value.
  * The runtime system claims a REGRET BOUND, not victories: within
    1-3% of the per-trace oracle static without workload knowledge, at
    ~zero toggle cost, with the engine determinism work (whole-chain
    graph, prefill skip) as standalone contributions.
  * SCOPING LAW (measured + predicted): intra-column switching ceilings
    are small -- +5.9% max over ALL workload compositions on the 8B
    column, +10.4% on 32B (transfer-estimated) -- so runtime switching
    pays only where regimes span columns/content with large deltas.
    Report as a boundary result, not a limitation buried.
  * RL note: the drift model REFUTES the RL switching-win hypothesis
    (-2.0%: monotone drift is tracked by one static). The RL argument
    that stands is trained-draft STALENESS -> the training-free lever
    family; do not stage the RL demo as a switching win.
- Sec D (eval): normal serving cells (already measured: 1.91x/2.77x
  headlines) + RL-rollout trace (NEW), with the staleness argument.
- Honest ledger carried (retractions, corrections, one-box scope until
  the hardware column lands).

## Measurement queue for the paper (priority order)

1. ~~82-E0 toggle-cost table (incl. CPU KV-requant path)~~ DONE
   2026-07-18 — Sec C's core (results_e0.md).
2. Second-hardware R column (91-min protocol) — validates axis 2.
3. RL-rollout-style trace eval — Sec D, staged as the STALENESS /
   training-free-tracking argument (NOT a switching win: the drift
   model predicts -2%).
4. ~~Switching demo on a regime-shifting trace~~ RUN 2026-07-18
   (82/results_e2.md): FAIL-HONEST on an OFF-heavy real-data trace --
   omniscient switching ceiling was only +1.7% over static-OFF (the one
   spec-favorable regime carried 9% of tokens), so no detector pays for
   itself there; policy beat both spec statics and held 98.5% of AR in
   OFF regimes while armed. The dwell-time law is now MEASURED: with
   toggle cost ~0, detection is the binding constraint and the win
   condition is spec-favorable regime VOLUME. Sec C presents the
   machinery + the law. SUPERSEDED CODA 2026-07-19: the full demo
   program (E2/E2b/E2c + wholechain stack + compiled policy, 82/
   results_e2.md) landed at regret 1-3% with an E2b win on every
   variant; the gap prediction shows this is near the intra-column
   ceiling (+5.9% max at 8B). 32B runtime trace RUN 2026-07-19: the
   +10.4% prediction was FALSIFIED (it rested on harness-transfer
   cells; the deployment compile priced b1/14k at ~0.99 vs the
   harness's 1.63) -- oracle ceiling +1.9% over best static, policy
   regret -1.4%. Scoping law confirmed at both scales; the transfer
   failure is itself Sec B evidence for measure-on-deployment.

## Two-paper fallback (recorded, not active)

If Sec C/D outgrow the page budget: paper 1 = current draft + hardware
column; paper 2 = switching + RL with paper 1 imported. Decision point:
after the toggle table and one switching demo exist.
