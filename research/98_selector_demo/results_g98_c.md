# G98-C (W98-R2): the corrected model, scored — 47/48 covered

First record of the completed Round-2 campaign, 2026-08-14. Numbers are from
`data/g98_c/round2_result.json` (v9); interpretation against the
preregistered claim belongs to the researcher's read of `w98r2_prereg.md`
and is not asserted here.

## Verdict numbers

The registered claim has two clauses; both are now scored.

* **Zero false eliminations — and the rule was EXERCISED.** Under
  `(K+1)/q_lo < 1.015` with the most elimination-favorable defensible `q`
  (optimistic draft bound inside the full measured armed step, over
  verify-only), the sound rule fired **5 times** — `target-matching/w1024/
  skip8` at R1 and R8, `target-matching/w128/skip4` at R1, R6, R8; min
  `s_max` 0.902 — and in every case the measured `q` confirms the kill
  (`d1p_false_elimination.json`, `score_w98r2_elimination.py`). Round 1
  reported this rule NOT EXERCISED for lack of a scored surface; Round 2
  both covers and correctly eliminates. The coverage-miss cell is not among
  the eliminated, so the clauses do not interact.
* **Coverage: 47/48 (97.9%)** of held-out (cell, regime) pairs inside their
  committed intervals.
* **All six regimes resolvable** — including R5 and R5cot, which Round 1
  registered as NOT RESOLVABLE on piecewise+Marlin.
* Prediction quality: median |rel. error| **0.53%**, max 3.96%.
* The single miss: `w4a16-quantized/w512/skip16` at R1 — measured 11.917 ms
  against interval [11.292, 11.880] ms, an overshoot of the upper bound by
  0.3% (37 us) on the most composed cell at the most host-exposed regime.
* Fitted `F` at R1: **3.256 ms** against the 3.66 ms registered from
  checkpoint bytes — relevant to the section-7 question of what the fitted
  floor absorbs (the +2.4 ms profiler-sync cost is present in both rounds'
  measurements by design).

## Where and how it ran

On **h104 (bare metal)**, GPU 7 / NUMA node 1, after the relocation and
infrastructure chain recorded in `results_clamp_investigation.md` sections
10-11: v7 (box relocation, compile-cache and PTX toolchain fixes), v8 (lane
GPU 4 -> 7 around a co-tenant), v9 (stage readers exclude v6 telemetry
sidecars — a latent bug no prior campaign survived long enough to hit).

**Every one of the 35 boots passed both gates on its first attempt** — 6
anchors_pre + 15 fit under v8 (08:52-09:31), then resume under v9: fits and
committed predictions at 09:35, 8 held-out cells (09:35-09:50), 6
anchors_post, scored 10:00:56. The X29 sentinel ran on GPU 5 throughout;
its record (`data/probe_flip_sentinel/channels_h104.jsonl`) is the
box-state evidence that the window was clean.

Contrast the box this campaign was driven off of: 40+ gated attempts on the
QEMU/KVM guest without a single accepted cell. The clamp investigation and
its verdict instrument are `results_clamp_investigation.md`; the remaining
old-box step (X29 there, one clamped + one clean burst) is unchanged.

## Order of operations (barrier integrity)

Anchors and fit cells were measured and accepted before `d1p_predictions`
was committed (09:35), and every held-out measurement postdates the commit.
The v8->v9 re-issue between fit measurement and scoring changed only the
stage readers' file filtering; cell records are create-only and were never
re-measured (`resumable` policy), so the falsifiability barrier holds
across the re-issue.
