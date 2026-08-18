# W100 amendment 1: the identity gate is unsatisfiable as registered

Authored 2026-08-17, after the `off b8` boot landed and **before any
scoring**. Supersedes rule 3 of `w100_prereg.md` and amends the metric
section. Everything else in the v1 registration stands. The off-b8
violations remain in the boot records as the evidence that forced this
amendment; they are reinterpreted, not erased.

## What happened

Rule 3 required every T=0 arm to reproduce the stock arm's outputs
bit-exactly, justified by losslessness. The first spec-engine boot
(`off`, K=0 — target-only decode) diverged from stock on 9/16 SS
requests and 16/16 on every long cell.

## Why it is numerics, not a bug

* SS (~180 out tokens): **7/16 bit-identical**. A systematic engine bug
  spares nothing; scattered greedy tie-flips spare 44%.
* Identical-fraction falls with output length exactly as a compounding
  per-token flip rate of ~0.5% predicts: (1-p)^180 ~ 0.44 at SS,
  (1-p)^700 ~ 0.03 at LI (0/16 observed identical).
* Per-request length deltas are two-sided and near-balanced
  (LI net +225 tokens over 16 requests; LIO mixed signs), including two
  same-length-but-different-text requests — single flip, then recovery.
* LO shows chaotic amplification: one flipped token re-routes a whole
  reasoning path (deltas from -17,893 to +21,359).

The registered gate assumed bitwise batch-invariance across engine
configurations. vLLM does not provide it: the spec-decode scheduler
composes batches and kernel launches differently from the stock engine,
logits differ in low bits, and greedy argmax flips on near-ties.
Losslessness guarantees distribution-preservation, not bit-equality
across engine configs. The v1 gate was a design error. (X32 could never
see this: `ignore_eos` + fixed budgets made content divergence
invisible.)

## Amended rules

**3a. Length-sanity gate** (replaces exact identity across engine
configs): per (cell, batch, arm) at T=0, against stock —
mean output tokens within ±10% (LO exempt from the mean bound, chaotic
amplification), and a two-sided sign test on per-request deltas: fail if
more than 80% of nonzero deltas share a sign on cells with n >= 16.
This still catches the bugs the identity gate existed for (a lever
bleeding into the target path truncates or inflates systematically);
it no longer fails on symmetric tie-flip jitter.

**3b. Divergence diagnostic** (reported, never gated): identical-output
fraction and implied per-token flip rate per cell, from the stored
sha256s.

**3c. Exact identity is retained** as a gate only where bit-equality is
actually guaranteed: same engine config, same batching — i.e. re-boots
of the SAME arm (phase-98 boot-determinism checks). It is no longer
applied across engine configs.

**Metric amendment.** Wall-clock drain ratio remains the headline for
LI/LIO/SS, where length jitter is ±2% and enters the ratio at first
order as noise smaller than boot variance; the per-output-token time
ratio is reported alongside. For **LO** the drain ratio is NOT a valid
score (token counts differ by up to ±20% between arms and the marginal
tokens are the expensive late-context ones): LO is scored on
**context-normalized per-token time** — per-token time corrected by the
token-weighted mean context position of each arm's own lengths, with
the correction estimator defined once in the analysis script and
applied identically to every arm, stock included. The uncorrected
ratios are reported next to it.

## Governance

Barrier v2 (`w100-final-grid-barrier-v2`) pins this amendment, the
updated `w100_protocol.py`, and the v1 artifacts. The running campaign
was not modified mid-flight: its orchestrator still logs v1 identity
violations, which the analysis re-scores under 3a/3b. No measurement is
affected by the amendment; only gating semantics and the LO scoring
rule change.
