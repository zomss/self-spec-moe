# Phase 90 — hierarchical lever search: proxy-scored heterogeneous
# configs + the C2 hardening

Source: user direction 2026-07-23 (STATUS_CONTRIBUTIONS discussion).
The current pool is coarse (global, per-family settings); the real
space is per-layer/per-head heterogeneous (window/sinks per layer,
pruning fraction per layer/head) — hundreds..thousands of configs.
The scheme to establish (C2's algorithmic content):

  stage 1: ANALYTIC cost x measured per-(hw,kernel,shape) realization
           factor -> latency-envelope prefilter (no per-config profiling)
  stage 2: IMPORTANCE-PROXY scores (one forward pass over task refs)
           -> shortlist -> direct beta on shortlist only
  stage 3: per-regime pools -> switch-cost-aware non-stationary
           bandit at runtime (formalizes the existing argmax/EMA/
           hysteresis/probe machinery)

First target: the R4 summarization headroom. Measured facts this
phase attacks: uniform window accept is a STEP function (win2048
gains nothing, win8192 jumps accept 2.70->4.06) but full-window chain
prices out at R~1.0 (attention-bound, both kernels). Hypothesis: a
few retrieval-heavy heads/layers carry the beyond-window attention
mass (duo-attention finding); giving ONLY them full context captures
most of the accept gain at near-win512 cost. Per-layer windows are
attention METADATA -> E0-class free toggles -> bandit-compatible.

## Pre-registered predictions (before measurement)

P1 (proxy validity): per-layer/head attention-locality scores
   computed on task refs rank-correlate with the COMMITTED beta
   record (77/83/86/87 CSVs: singles, frontiers, 3 scales) at
   Spearman >= 0.7 for window-family arms; importance-score ranking
   reproduces the iterative-greedy skip sets' ORDER (83/86
   iter_greedy csvs) at both 8B and 32B.
P2 (concentration): <= 25% of (layer, head) pairs carry >= 70% of
   the beyond-512 attention mass on CNN/DM refs at 8B.
P3 (hetero-window accept): full-ctx on the P2-selected heads +
   win512 elsewhere recovers accept >= 3.6 on R4 (uniform win8192
   = 4.06 ceiling, win512 = 2.70 floor).
P4 (e2e): R4 hetero-window arm >= 1.2x vs AR (from 1.08x Hum-K2),
   with draft chain R <= 1.3x the win512 chain cost.

## Plan

- E1 (proxy validation, GPU-light): score computation over task refs
  (attention-mass-beyond-window per layer/head from ONE forward pass;
  angular-distance importance per layer) -> rank-correlate against
  every committed beta arm that varies the corresponding axis. Gate:
  P1. If P1 fails, the proxy stage is dead -> fall back to
  leave-one-out (L-runs) screening; scheme survives with a costlier
  stage 2.
- E2 (locality profile): per-head beyond-window mass on CNN/DM 8k
  refs (R4's data), Qwen3-8B; emit the retrieval-head set. Gate: P2.
- E3 (hetero-window beta): 77-harness with per-layer/per-head window
  masks (HF-proxy side first — no engine surgery needed for beta);
  measure accept at {selected-heads-full + win512 rest} vs uniform
  512/2048/8192. Gate: P3.
- E4 (e2e, build-on-selection): only if P3 passes — fork support for
  per-layer draft windows (VLLM_SELF_SPEC_DRAFT_KV_WINDOW_MAP;
  scratchpad currently assumes uniform cap 544 -> per-layer
  scratchpad sizes = the real surgery), then the R4 canonical arm.
  Gate: P4.
- E5 (bandit formalization, parallel track): replace the hand-tuned
  argmax+hysteresis with sliding-window Thompson/UCB over the
  per-cell pool; must encode measured switch costs (arm 2%, disarm
  free, b1 min-batch SNR gate, probe budget). Backtest on the
  committed E2/E3 traces (82/89 jsons) BEFORE any live run — the
  eager-gate anti-pattern (0.89x) is the cautionary baseline.
- Cost accounting side-quest (C2 work item 1): assemble the
  cost-to-onboard table from committed logs while E1 runs.

## Constraints

GPUs 0,1,6,7 (per-request grants otherwise); caches /data;
.venv/bin/python; disjoint-artifact rule (nothing scored on its own
construction refs); build-on-selection (E4 only after P3).

## Artifacts

results_hier.md; data/proxy_scores*.csv, locality_q3_8b.csv,
beta_hetero.csv, rankcorr.json; scripts/e1_proxy_validate.py,
e2_locality.py, e3_hetero_beta.py.
