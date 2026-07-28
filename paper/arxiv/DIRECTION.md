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
- Sec B (search) — REWRITTEN 2026-07-24 after phase 90: the claim is
  a measured THEOREM, not a recipe preference. "Acceptance-importance
  cannot be scored, only measured — on-policy, conditionally, at
  full-ref resolution." Supported by the five-way proxy falsification
  (T11 #9: offline geometry INVERTED -0.77; margin-Taylor inverted
  off-policy; binary/continuous on-policy subsampling both ~0.35;
  learned predictor 0.08-0.22 with the data-circularity argument),
  the non-additivity control (perfect singles score -0.09 on set
  growth), and TWO mechanism discoveries: (a) ref-distribution
  dependence (off-policy scoring inverts rankings — the E1d sign
  flip), (b) collectivity (window-need is super-modular: partial
  relief fails in ctx-space AND layer-space; 90-E3 controls beat
  measured-ranked sets). Corollaries: compression-literature
  importance does not transfer to draft construction
  (perplexity-importance != acceptance-importance, measured); and
  KnapSpec's additive-cosine knapsack sits in the refuted class —
  the mechanistic explanation for our h2h margin. The amended
  three-stage scheme: analytic cost x measured kernel factor ->
  conditional on-policy beta (proxies only as round-ordering priors,
  0.3-0.4 grade; cross-model transfer 0.43 as cold-start) ->
  per-regime pools + switch-cost-aware adaptation. Cost: T10 —
  zero-to-compiled-policy ~2-4 GPU-hours/column; 91-min hardware
  protocol. Audit loop as first-class method component (T11: 5
  refutations -> 5 corrections). Literature: field converged to
  measurement-based selection (SWIFT/CLaSp/KNN-SSD/BanditSpec cited;
  Draft&Verify BayesOpt as the direct-measurement ancestor).
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
    are ~+10% over ALL workload compositions (8B +10.4% on the
    fixed-skip cells -- doubled from +5.9% once the skip bug stopped
    throttling spec-win cells; 32B +10.4% oracle-est, measured trace
    ceiling +1.9%) -- vs +39.1% cross-column. Runtime switching pays
    mainly where regimes span columns/content with large deltas.
    Report as a boundary result, not a limitation buried. Fixed-skip
    revalidation (82/results_e2.md): the policy WINS 2 of 3 traces
    outright (E2 979.2 vs best static 976.6; E2b 817.8 vs 799.5) and
    is -2.3% on the third -- one config vs three different per-trace
    static winners.
  * RL note: the drift model REFUTES the RL switching-win hypothesis
    (-2.0%: monotone drift is tracked by one static). The RL argument
    that stands is trained-draft STALENESS -> the training-free lever
    family; do not stage the RL demo as a switching win.
- Sec D (eval): decode-cell headlines (1.91x/2.77x; W4A8-Humming
  2.19x) + the SERVING WALL headline (2026-07-19, T6): **1.80x wall
  aggregate / 1.90x at b16** vs AR on the full serving driver at
  reasoning shapes (14k-doc RAG + 3k-token CoT), S_dec 2.13 =
  decode-cell record reproduced in serving. Plus the RL-rollout trace
  (staleness framing).
- BEAT-ALL-BASELINES PROGRAM (user-set, steps 0-3; DONE 2026-07-21,
  phases 88-89 -- the paper's new coverage + adaptation spine):
  * Step 0/1 (T8): canonical regime suite (real prior-work datasets);
    the framework finds an AR-beating setting at **9/9 8B regimes**
    (shallow-K converts the two losses: accept is FRONT-LOADED in
    depth); 32B: 8/9 + winners split across kernel x depth x
    composition x OFF -- the lever-diversity evidence ON CANONICAL
    DATA (single-cell h2h understates diversity).
  * Step 2: KnapSpec closed -- Humming ~1.51x, and the TRIPLE
    composition (skip x W4A8-Hum x win512, their own lever included)
    ~**1.57x vs their 1.43x (+10%)**. Composition is scale-keyed:
    skip stacks at 32B (+3-6%, accept ~0 cost, third sub-additivity
    confirmation), priced out at 8B.
  * Step 3 (T9): DRAM lever swap MEASURED -- 113ms pinned swap,
    graph-replay bit-exact, live mid-serving refresh 114-116ms; the
    RL drift demo: staleness -45% cliff -> full system (per-step
    policy + detector-fired refresh) **1.055x over AR** at the map's
    thinnest cell. Selection accounting: oracle composites 1.26x/
    1.20x; switching bound +4.2%/+7.3% uniform-mix, CONCENTRATED
    (+7-33% per-cell) exactly where statics lose -- the RL regime.
  * Reframe note for Sec C: the RL demo IS now a switching-system win
    (depth argmax + OFF gate + refresh beat AR and every static);
    the earlier "-2% drift model refutes switching win" applied to
    lever-flip switching WITHOUT refresh -- the refresh axis is what
    converts drift from a threat into the system's home turf.
  * STAGE-3 FORMALIZATION CODA (2026-07-24, phase 91 / T12): the
    runtime controller is now formally grounded. A Thompson bandit
    (discounted Beta posterior over f, switch-cost-aware) was built,
    sim-validated (regret 5.1% vs the argmax's 8.2%), deployed live
    -- and lost (0.996x vs 1.042x), through three measured failure
    modes the sim missed (posterior granularity x batch; sampling
    variance at the arming boundary; DETECTOR STARVATION under
    censored feedback). The ablation ladder necessity-proves every
    hand-tuned mechanism of the deployed policy: it is the measured
    optimum, not an ad-hoc heuristic. Novel finding for the setting:
    exploration must be budgeted for DETECTION liveness (the drift
    detector consumes the same evidence the policy's disarm cuts
    off). Cite BanditSpec/Not-a-Bandit for regret theory;
    measure-on-deployment now demonstrated for controllers, not just
    configs. Marlin K-grid + multi-seed CI runs in flight (91).
- Honest ledger carried (retractions, corrections, one-box scope until
  the hardware column lands).

## Measurement queue for the paper (priority order)

1. ~~82-E0 toggle-cost table (incl. CPU KV-requant path)~~ DONE
   2026-07-18 — Sec C's core (results_e0.md).
2. Second-hardware R column (91-min protocol) — validates axis 2.
3. ~~RL-rollout-style trace eval~~ DONE 2026-07-21 (89/results_swap
   E3/E3b): staleness measured (-45% cliff), full system beats AR
   1.055x with detector-fired 113ms DRAM refresh — supersedes the
   "-2% no-switching-win" staging: with the REFRESH axis it IS a win.
   Remaining hygiene: Humming odd-width bug report upstream; direct
   harness confirmation of the anchored 32B Humming rows.
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
   FIXED-SKIP REVALIDATION 2026-07-19 (the final 8B record): with the
   scheduler bug corrected, spec statics surge where spec wins (E2c k4
   +31.7% over OFF; b16/14k cell 1.59x) and the POLICY WINS 2 of 3
   traces outright (E2 979.2, E2b 817.8) at -2.3% on the third --
   cite THESE numbers, buggy-era tables archived as *_buggyskip.

## Two-paper fallback (recorded, not active)

If Sec C/D outgrow the page budget: paper 1 = current draft + hardware
column; paper 2 = switching + RL with paper 1 imported. Decision point:
after the toggle table and one switching demo exist.
