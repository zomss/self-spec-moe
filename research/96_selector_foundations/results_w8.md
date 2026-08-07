# W8 results — constructive Round-1 cost model

Pre-registration: `w8_cost_model.md` (commit 3753f00dd, before any run).
Artifacts: `data/w8/` — W8a natural-EOS matrix (12 profiled boots),
W8b/W8c constant-batch re-measurement (fixed-256: 6 off + 6 profiled
uncond), W8d unperturbed uncond b32 (2 boots). Scorers `score_w8.py`
(W8a) and `score_w8_final.py` (final).

## 1. Why the protocol changed mid-arc (W8a → W8b/c)

W8a reconstructed MLA to 2.8%/4.6% but MoE to −53%/−56%. Not a model
failure: MoE's R2 decodes stop at ~200 tokens on natural EOS, so the
batch DRAINS (32→1) mid-run and both the e2e rate and the profiler's
mean step time average over a collapsing width. MLA's 2048-token
ceiling decodes keep the batch full, which is exactly why only MLA
survived. The e2e-derived T was inflated 1.9× (b8) and 2.3× (b32) for
MoE. Fix: `G93_FIXED_LEN` (ignore_eos + fixed max_tokens) pins the
width; both arches re-measured at fixed-256.

## 2. Measured cost terms (fixed-256, T from unperturbed off arms)

| cell | T (ms) | D/T | V | ovh/T | D vs roofline | V bytes-law |
|------|-------|-----|-----|-------|------|------|
| MLA b1 | 4.28 | 1.69 | 1.75 | 0.79 | **7.8×** | 2.37 |
| MLA b8 | 8.90 | 0.98 | 1.41 | 0.53 | **3.1×** | 1.65 |
| MLA b32 | 13.07 | 0.78 | 1.19 | 0.42 | **2.2×** | 1.04 |
| MoE b1 | 4.72 | 1.64 | 1.54 | 0.83 | **15.2×** | 2.81 |
| MoE b8 | 7.50 | 1.00 | 1.38 | 0.62 | **3.8×** | 2.14 |
| MoE b32 | 11.52 | 0.74 | 1.34 | 0.49 | **2.1×** | 1.14 |

**P-W8b CONFIRMED at b1, REFUTED at b32.** Predicted ≥2× roofline at
b1 (measured 7.8×/15.2× — far worse) and ≤1.3× at b32 (measured
2.1–2.2× — the gap never closes). The draft step is dispatch-bound,
not bandwidth-bound: MoE's 128 experts × 768 intermediate make
per-expert GEMMs at M=1 nearly pure launch overhead, which is why its
b1 gap is 2× MLA's. **This is the mechanism behind R > 1.**

**P-W8a REFUTED in magnitude, CONFIRMED in shape.** V(4,b1) = 1.75 /
1.54 (predicted ≥2); monotone decrease ✓; V(4,b32) = 1.19/1.34 (≤1.3
holds for MLA, misses for MoE). The i.i.d. coverage bytes-law
over-predicts V at low batch (2.37 vs 1.75; 2.81 vs 1.54) and
converges near saturation. Mechanism: below saturation the step is
latency-bound, so extra verify tokens do not cost their bytes; the
bytes law becomes valid exactly when the step becomes bandwidth-bound.

## 3b. P-W8c — NOW TESTED, and it PASSES (2026-08-07)

The circular check below stands retracted. But a genuine out-of-sample
test exists in the banked data and I missed it: W8d/W9 ran UNPROFILED
uncond boots at b32 under the same fixed-256 protocol. Profiled boots
were pre-registered as "not scored serving artifacts", so those
unprofiled runs are held-out data for the cost model.

Predict the unprofiled denominator from the profiled step times
(P = draft_chain + verify, T from the unprofiled off arm):

| cell | predicted τ* = P/T | actual τ/S (unprofiled) | error |
|---|---|---|---|
| MLA b32 | 4.714 | 4.643 (τ=4.939, S=1.0637) | **+1.5%** |
| MoE b32 | 4.798 | 4.813 (τ=4.467, S=0.9282) | **−0.3%** |

Both inside the pre-registered ±10%. And the ARM/OFF call follows
directly: MLA τ=4.94 > τ*=4.71 → ARM (actual S=1.064); MoE τ=4.47 <
τ*=4.80 → OFF (actual S=0.928). **P-W8c is CONFIRMED at b32 on both
architectures.**

### Consequence: the "instrument coverage" reading was wrong

§4 reported that the two profiled regions "contain 78–87% of the armed
step", implying they MISS real work. They do not. The regions predict
the UNPROFILED armed-step cost to within 1.5%; the missing 13–22% in a
profiled boot's own serving numbers is the profiler's sync overhead
sitting OUTSIDE the regions. So the pre-registered rule to discard
profiled serving numbers was not just cautious — it was exactly the
right cut, and the region times are sound Round-1 inputs.

This also softens §4's conclusion about Stage 0: the terms are not
25–29% inflated as cost inputs (that figure is the perturbation of the
profiled boot's END-TO-END rate, not of its region times), so the
Stage-0 deflation exercise in §4 was over-conservative. Stage 0's
b1/b8 OFF proofs stand on the region times directly.

## 3. P-W8c — the ORIGINAL check: NOT A TEST (retracted, T11)

`score_w8_final.py` first reported "coverage-corrected reconstruction"
PASS at −0.0% on all six cells. That is an algebraic identity:

    corrected = recon/coverage = (P/T)/(P/step_serv) = step_serv/T
              = (b·τ·1e3/rate_u)/(b·1e3/rate_off) = τ/S_ss = meas

P cancels, so it holds for ANY cost model. Equivalently recon/meas ≡
coverage (verified to 4 dp). I also reported these to the user as two
independent measurements agreeing to 1–2 points; they are the same
number. **Retracted.** P-W8c remains untested; a real test needs
out-of-sample prediction (e.g. predict K=2 or b64 from terms measured
elsewhere), which this arc does not have.

Surviving content: instrument coverage = 0.85/0.81/0.81 (MLA) and
0.87/0.82/0.78 (MoE) — the fraction of the armed step inside the two
profiled regions.

## 4. Instrument perturbation, measured (W8d)

Unperturbed uncond b32 at fixed-256 vs the same cell profiled:

| | profiled S | unperturbed S | perturbation | W7 natural |
|---|---|---|---|---|
| MLA b32 | 0.850 | **1.064** | 1.251× | 1.129 |
| MoE b32 | 0.720 | **0.928** | 1.289× | 1.087 |

The profiler's sync-bracketed regions inflate armed-step cost by
25–29%. Two consequences:

- The pre-registered "profiled boots are not scored serving artifacts"
  rule was necessary, and is now quantified.
- **The empirical Stage-0 is built on inflated terms.** Deflating by
  the measured factor: MLA b1 S_max 0.54→0.67, b8 0.85→**1.07**,
  b32 1.06→1.33; MoE b1 0.56→0.72, b8 0.84→**1.08**, b32 1.04→1.34.
  So Stage-0 proves OFF only at **b1** for both arches; b8 becomes
  undecided. (The factor is measured at b32 only; at b1 the shorter
  steps likely carry a larger relative perturbation, so even the b1
  proof is not certified.)

## 5. Stage-0 verdicts

- **Theoretical floor (HBM roofline at 3.35 TB/s, V≥1, ovh≥0):
  UNDECIDED everywhere** — S_max = 2.07–3.49 > 1 in all six cells.
  **P-W8d REFUTED in its strong form**, and this is the arc's most
  useful result: pure cost theory cannot eliminate a single cell,
  because the roofline-to-achievable gap is 2–15×. Offline profiling
  is a NECESSARY Round-1 input, not a refinement — the user's
  proposal (1) is vindicated by its own failure mode.
- **Empirical floor**: proves OFF at b1 (both arches) after deflation;
  b8 undecided; b32 survives. The b8/b32 OFF/ARM verdicts continue to
  rest on W7's unperturbed serving measurements, not on Stage-0.

## 6. Open discrepancy — MoE b32 is protocol-sensitive

MLA b32 arms under both protocols (1.064 fixed-256, 1.129 W7). MoE
b32/R2 arms under W7 (1.087) but LOSES unperturbed at fixed-256
(0.928). τ is nearly identical (4.47 vs 4.43), so this is a cost-side,
not acceptance-side, difference. Two uncontrolled confounds between
the protocols: (a) natural-EOS drain vs constant width, (b)
`max_num_seqs` = 64 (W7's multi-batch sweep) vs 32 (single-batch
boots), which changes CUDA-graph capture sizes and spec token-slot
reservation. **W7's verdict stands as the deployment-relevant one**
(it used the arc's standard serving protocol); the P-W7d refutation is
now qualified as protocol-dependent and needs one controlled boot
(fixed-256 at max_num_seqs=64) to isolate. Recorded, not resolved.

## 7. What W8 delivers to the design

The mechanism for the C2 dichotomy's R side is now measured, not
inferred: **R > 1 at low batch is dominated by kernel/dispatch
inefficiency (2–15× off roofline), with verify-coverage excess second
(+19–75%) and stack overhead third (0.4–0.8 target-steps per armed
step)**. Round 1 can be made constructive, but only with an
unperturbed profiler — CUDA events or nsys rather than sync-bracketed
regions — which is the concrete next engineering step.
