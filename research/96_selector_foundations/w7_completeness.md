# W7 — MLA/MoE Round-2 completeness pass (pre-registration)

Written BEFORE any W7 run is scored (W3 doctrine). Purpose: close the
last measurement gap in the two-round design — Round-2 confirmation of
the Round-1 {OFF} tie-sets for the two architectures whose story so far
rests on banked physics only (phase-94 C2 oracle tables), plus a live
verdict on the single candidate cell (b32).

## Why the phase-95 E3 artifacts cannot serve

`95/data/e3_mla_{off,uncond}_s0.json` predate the notune protocol:
autotune lottery armed (W1: 40% boot variance), ITERS=3 (episode
rejection needs ≥4), one boot (no cross-boot certification), and the
MoE arms never ran at all. The MLA b1 round spread [120.6, 130.3,
143.7] is exactly the W1-diagnosed noise. They are retained as a
pre-notune prior, nothing more.

## Protocol

- Runner: `93/scripts/run_grid.py` with the new `G93_TUNE=0` plumb
  (kernel_config `enable_flashinfer_autotune=False`), same env stack as
  `95/run_e3_gate.sh` SHARED (the stack the banked R tables were built
  under). ITERS=4, batches {1,8,32,64}, regimes {R1,R2,R6}, seed-0
  content (single draw, disclosed — same as E3; cross-boot
  certification needs content identity, not diversity).
- Arms: `off` (AR anchor) and `uncond` (map-best static config, K=4,
  no policy file). The old `gated` arm is NOT rerun: it realized the
  two-compensating-errors selector this phase retired; the gate verdict
  follows directly from off/uncond S.
- MLA (DeepSeek-V2-Lite, TP1, ceiling 2048, draft w8chan): two parallel
  single-GPU lanes; lane-g boots off then uncond on GPU g. The two
  certification boots per arm land on DIFFERENT GPU lanes by design —
  every suppressed cell this arc sat on the GPU0 lane, so lane
  disagreement itself localizes an episode.
- MoE (Qwen3-30B-A3B, TP2 on GPUs 0+1, ceiling 16384, draft w4a16):
  four sequential boots — off ×2, uncond ×2.
- Scoring: per cell, pool the 8 rounds (2 boots × 4 iters), reference =
  2nd-highest round, reject rounds >5% below reference; certified iff
  each boot retains ≥2 rounds and the boot means agree within 2%.

## Pre-registered predictions

- **P-W7a (MLA below b32)**: uncond S_dec < 1 in every certified
  b1/b8 cell. Strong form: banked R ≥ 1.21 puts break-even
  τ* = 4R+1 ≥ 5.85 above the K=4 maximum τ = 5 — no acceptance
  behavior can flip these cells. Round-1 soundness made concrete.
- **P-W7b (MoE below b32)**: same, R ≥ 1.14 → τ* ≥ 5.55 > 5.
- **P-W7c (MLA b32, the candidate)**: oracle S_ref = 1.02 (K2) /
  1.04 (K4). Decision rule, fixed now: the arm is banked iff certified
  mean S_dec ≥ 1.02 (arming rent 1.5% + noise floor) on BOTH boots;
  anything less is recorded as a tie → OFF (fail-closed, the llama
  doctrine). Prediction: MARGINAL — we do not predict the side.
- **P-W7d (MoE b32)**: uncond S_dec < 1 (oracle 0.94–0.99) → OFF
  everywhere; the MoE gate story closes with zero armed cells.
- **P-W7e (b64)**: no C2 row exists (map completeness only). Report
  measured S; no acceptance-independent bound is available, so no
  arming decision may be made from this pass at b64.

## Disclosed caveat — V2-Lite τ inflation

V2-Lite loops at T=0 on 8/9 datasets (ceiling protocol, 93): measured
τ ≈ 5.0 on ceiling-clipped loop content is an UPPER bound on
real-content τ. Below b32 this is irrelevant (verdict is
acceptance-independent). At b32 it biases TOWARD arming: an OFF verdict
is a fortiori sound; an ARM verdict carries a content-dependence flag
(the llama lesson) and would need real-content confirmation before
deployment.
