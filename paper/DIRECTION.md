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

## Single-paper structure (foundation = research/79_paper/paper_draft.md)

- Sec A (foundation, compressed from current secs 3-6): the two surfaces,
  the taxonomy-complete pool, the realization-priced model, the LCB
  selector + audit loop. The current 9.4k-word draft compresses to
  ~40% here; maps become the INPUT to the system, not the headline.
- Sec B (search): the corrected recipe vs profile-then-solve, with the
  KnapSpec head-to-head (our backtest's m=0 row IS their recipe) and the
  91-min protocol.
- Sec C (switching system): toggle-cost table (NEW measurement), the
  dwell-time law, the regime detector (batch/ctx/content), policy = the
  map + hysteresis; delivered switching demo.
- Sec D (eval): normal serving cells (already measured: 1.91x/2.77x
  headlines) + RL-rollout trace (NEW), with the staleness argument.
- Honest ledger carried (retractions, corrections, one-box scope until
  the hardware column lands).

## Measurement queue for the paper (priority order)

1. ~~82-E0 toggle-cost table (incl. CPU KV-requant path)~~ DONE
   2026-07-18 — Sec C's core (results_e0.md).
2. Second-hardware R column (91-min protocol) — validates axis 2.
3. RL-rollout-style trace eval (batch drain + on-policy text) — Sec D.
4. ~~Switching demo on a regime-shifting trace~~ RUN 2026-07-18
   (82/results_e2.md): FAIL-HONEST on an OFF-heavy real-data trace --
   omniscient switching ceiling was only +1.7% over static-OFF (the one
   spec-favorable regime carried 9% of tokens), so no detector pays for
   itself there; policy beat both spec statics and held 98.5% of AR in
   OFF regimes while armed. The dwell-time law is now MEASURED: with
   toggle cost ~0, detection is the binding constraint and the win
   condition is spec-favorable regime VOLUME. Sec C presents the
   machinery + the law; the RL trace (item 3) is the win stage
   (content drift + long-decode batches).

## Two-paper fallback (recorded, not active)

If Sec C/D outgrow the page budget: paper 1 = current draft + hardware
column; paper 2 = switching + RL with paper 1 imported. Decision point:
after the toggle table and one switching demo exist.
