# Phase 98 G98-0 — CPU harness

Status: **PASS for the harness portion: accounting closure, u-binned
position-resolved acceptance, factored cost model, and knapsack identity
selection are implemented and proven on synthetic data. The prompt-manifest
freeze remains open and belongs to `w98_prereg.md`. No GPU command was run
or authorized.**

Date: 2026-08-10.

## What was built

Three phase-local modules under `scripts/`, no engine imports, stdlib only:

- `w98_accounting.py` — `StepRow` validation, `E + C = A + H` closure with
  an optional independent frontend check, right-open u-bucket binning with
  a structural uniform-KMAX contract (mixed-K streams are rejected, clipped
  armed steps are excluded from counters and reported, unarmed steps count
  toward H only), `tau(k)` for every `k <= kmax` from one stream, and
  request-level percentile bootstrap for per-bucket `tau` and
  paired-by-request `delta tau` (identical request-id sets enforced).
- `w98_cost_model.py` — affine `T(Q)`/`P(Q)` fits with leave-one-out
  certification at the 5% W14 tolerance (never certified below three
  points), `q = P/T` required-acceptance labels, the factored composition
  model `D ~= keep_frac * (Wb*kappa_w + KVb*kappa_kv + c0)` fit from
  single-lever points by exact 3x3 least squares with a symmetric
  log-residual envelope (default 2x inflation for composed predictions),
  the sound elimination rule `(K+1)/q_lo < 1 + epsilon_arm`, and the
  skip-count ladder against a sound `tau_bar`.
- `w98_knapsack.py` — retention validation, top-k (uniform-cost knapsack),
  exact DP knapsack over integer-scaled heterogeneous costs, the admit-only
  product bound, random-set and worst-set controls, greedy forward
  selection as the escalation path, and the cross-regime consolidation
  matrix with the robust-single-set test.

## Validation

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  research/98_selector_demo/tests -q

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research/98_selector_demo/scripts/validate_w98_harness.py
```

The suite passes 56 tests. The validator reproduces eight registered
deterministic proofs and writes
`data/g98_0/w98_harness_validation.json`:

| proof | content |
| --- | --- |
| closure_identity | E + C = A + H holds; corrupted frontend count rejected |
| tau_recovery_all_depths | known chain-acceptance p recovered within 2% at every depth from one KMAX stream |
| u_binning_right_open | bucket edges are right-open bounds |
| mixed_k_rejected | a stream mixing proposal depths fails closed |
| factored_model_exact | planted kappas recovered; composed triple predicted exactly from singles |
| elimination_soundness_sweep | 500 randomized cases: every fired elimination is truly below 1 + epsilon_arm |
| knapsack_matches_brute_force | DP equals exhaustive argmax on heterogeneous costs |
| greedy_escalation_beats_bound | greedy sidesteps a planted pairwise conflict the product bound cannot see |

Ruff check and format pass all Phase 98 Python files. Key negative proofs in
the suite beyond the validator: singular factored designs are rejected,
curved latency data fails affine certification, paired bootstrap requires
identical request-id sets and returns exactly zero on identical streams,
infeasible knapsack targets are rejected, and conflicting regimes yield no
robust single skip set while agreeing regimes do.

## Deliberate semantics worth recording

- The uniform-KMAX contract makes mixed-K pooling a structural error, not a
  scoring-time correction.
- Clipped steps stay in the closure (via `C`) but never enter position
  counters, so `tau` is measured only on unclipped chain evidence while
  `tau_eff = E/H` remains the committed-work quantity.
- The factored model divides out `keep_frac` before fitting, so skip enters
  as physics and the fit only estimates the two byte-pool rates and the
  dispatch constant.
- `eliminate()` consumes a *lower* bound on `q`; the soundness sweep
  verifies the implication, not a point estimate.

## Open items for `w98_prereg.md`

- Prompt-manifest freeze (regimes, seeds, exact token IDs, hashes) — the
  remaining G98-0 item.
- Final u-bucket edges are a scored output of the first Round-2 run; the
  harness takes them as an input parameter.
- D1/D2/D3 matrices, tolerances, and held-out splits.

No GPU command is authorized by this record.
