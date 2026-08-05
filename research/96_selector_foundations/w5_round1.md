# W5 — Round 1 redesigned: tie-sets under measured uncertainty, and the W6 prediction

Pre-registered procedure, committed before any number below is computed.
Inputs are all committed artifacts: the C2 oracle pairs (validated by
W4b at 1.00×), the 93 off anchors, W2's per-regime live accepts, and the
W1/W4 constants.

## 1. What Round 1 is now

Per (arch, regime): a **tie-set** of (composition, K) candidates with
uncertainty intervals — not an argmax — plus the boot-class choice and
the runtime pool it implies. "The search must express 'cannot tell'"
(I3) is implemented as interval dominance: a candidate is eliminated
only when some other candidate's LOWER bound exceeds its UPPER bound.
OFF is always a member (S = 1 − parked(cell); parked = 0.6% at b≥16,
else 0).

## 2. Prediction model and uncertainty envelope

`S_pred(comp, K, regime) = (1 + f̂·K) / (K·R̂ + 1)` in **S_dec** (W3
currency), with:

- **R̂**: oracle mean R at the regime's mapped cell, inverted exactly as
  `compile_from_c2` (tau/S_ref identity). Band (all measured):
  - replication half-spread between the two oracle runs,
  - ±1% noise floor (W1),
  - **one-sided N-bias**: serving-length R is 0.88–1.00 × compile-N160 R
    (W4c: N160→serving-512 dropped R 6–12% within-realization),
  - R8's b16 cell: log2-linear interpolation over b∈{1,8,32}; the
    interpolation residual measured by predicting b8 from {b1,b32} is
    added to the band.
- **f̂**: separability transfer `f̂(regime, comp) = f_C4(comp) · ρ(regime)`
  where `ρ(regime) = f_live(regime, deployed comp) / f_C4(deployed
  comp)` — validated in §3 before use; its measured spread widens the f
  band. Where separability fails, the sound bounds replace the point
  estimate (`f_comp ≥ ∏ f_i` admits; `≤ min(f_i)·1.03` eliminates — S1).
- **Realization**: R̂ is scratchpad-realization. Long-decode regimes
  (≥512 expected tokens: R5cot, R8, R1, R6-ish) may use it for piecewise
  unchanged (W4c: <2% equivalence). Short-decode regimes (R5's ~72-tok
  answers, R4's 512-max summaries) carry the piecewise default and a
  widened band; the scratchpad is EXCLUDED from their tie-sets unless
  Round 2 measures it (the −15% cliff, W4c).

Regime→cell map (S_dec basis): R4→(8,8000); R5,R5cot→(8,14000);
R1→(1,2000); R6→(32,2000); R8→(16,2000) interpolated.

## 3. Separability validation (pre-registered test)

For every (arch, regime, window) where BOTH a live accept (W2/W4c) and
an oracle accept (same comp, K4) exist:
`ρ(regime, w) = f_live / f_C4`. **Separability holds iff, per regime,
ρ varies across windows by < 5% relative.** Holds → f̂ uses ρ with its
spread as the band. Fails → sound-bounds fallback for cross-composition
f, and the failure is reported (it would itself be a finding: content ×
composition interaction in acceptance).

Known caveat, disclosed: W2's live accepts are policy-mixed (K2/K4
blend), biasing accept DOWN vs pure K4. ρ is therefore a lower-bound
flavored estimate; the band carries it.

## 4. The W6 go/no-go prediction (the W3 §5 gate, applied)

- **B** = best single static (comp, K) by mean over regimes of
  `max(S_pred, 1)` (per-regime oracle OFF gate included), chosen
  exhaustively — the baseline gets every advantage.
- **T** = per-regime argmax of `max(S_pred, 1)` over that regime's
  tie-set.
- Primary read: point estimates. Robustness read: T at LOWER bounds vs B
  at UPPER bounds.
- **Verdict rule (binding, from w3_preregistration.md): build W6 only if
  mean(T − B) ≥ +2.0% S_dec, or one regime ≥ +5.0% with mean ≥ 0.** If
  even the OPTIMISTIC read (T upper vs B lower) fails, W6 is dead on the
  spot; if only the pessimistic read fails, W6's first act is to shrink
  the intervals on the deciding cells.

Scope: dense + llama (regime data exists). MLA/MoE keep their banked
gate story; their tie-sets are emitted for completeness from oracle data
only, flagged no-live-f.

## 5. Outputs

- `data/w5/tiesets_{arch}.json` — per regime: candidates with S intervals
- `results_w5.md` — separability verdict, tie-set sizes, capture-budget
  count, and the W6 prediction with the gate verdict
