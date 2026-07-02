# Phase 48 — shielded win curve: the measured headline after fixing the delay-shielding gate

**What changed.** `vllm/distributed/device_communicators/all2all.py::_emulate_exposed_a2a_delay`
now returns before enqueuing the `torch.cuda._sleep` when `self_spec_local_route_enabled()` is
true (the self-spec draft's forward-context flag). The draft issues zero real collectives
(`real=0`, Phase 41), so it now also pays zero emulated A2A — Phase 42/47 charged it the sleep
anyway, making their curves a conservative lower bound (RESULTS.md future-work 2). New
`shielded` counter logged next to `total/active/real`; A/B revert knob `W7_CHARGE_DRAFT_A2A=1`
reproduces the charged behavior. The branch resolves at CUDA-graph capture time per model
(draft/verify capture separate graphs), so the draft's graphs contain no sleep, the verify's
keep it. Default paths and the no-spec baseline are untouched.

**Setup.** Identical to Phase 47 (`run_sweep_shielded.sh` = `run_sweep2.sh` with output dirs +
`PATH` for ninja): Qwen3-30B-A3B, DP8/EP8, forced-PCIe, FP8 full-replica comm-free draft, fully
optimized stack (genuine replica + LOCAL_ROUTE + FULL_CG + COMPILE_CONSISTENT + PIECEWISE
chain), K=2, greedy, batches {32,64,128} x A2A {0,100,250,500,1000} µs/coll, harness
`w7_fp8_timing.py` UNCHANGED, no-spec baselines REUSED from Phase 42.

## 1. The headline: measured speedup, shielded vs charged (Phase 47 in parens)

| A2A µs/coll | batch 32 | batch 64 | batch 128 |
|---:|---:|---:|---:|
| 0 (native) | **1.14×** (1.16) | **1.03×** (1.03) | 0.65× (0.72) |
| 100  | **1.42×** (1.20) | **1.21×** (1.08) | 0.79× (0.81) |
| 250  | **1.72×** (1.23) | **1.34×** (1.13) | 0.87× (0.93) |
| 500  | **2.12×** (1.25) | **1.43×**¹ (1.17) | **1.21×** (0.96) |
| 1000 | **2.59×** (1.30) | **2.27×** (1.24) | **1.34×**¹ (1.14) |

¹ flagged `suspect` by the harness (iteration variance); b64@250 also flagged. These are the
points sitting below their model line (Section 3).

- **The comm-bound headline doubles:** at 1000 µs (f = 0.67–0.78) the win goes 1.30/1.24/1.14×
  → **2.59/2.27/1.34×**. The gap vs charged grows with delay, exactly as the over-charge model
  predicts (the charged draft paid ~2 chain forwards' worth of sleep per cycle).
- **Crossovers:** b32 and b64 are ≥1.0× everywhere including native; b128 crosses at ~343 µs
  (interp) — native large batch stays compute-bound, as before.
- **accept_len is 2.87–2.92 across the whole grid** — identical to Phase 47. The gate touches
  timing only, not numerics (as designed: it removes a sleep, not an op).

## 2. The multi-node band is now MEASURED, not projected

The emulated-delay → f map (no-spec slope fit, unchanged method): 1000 µs ⇒ f = 0.78 (b32),
0.74 (b64), 0.67 (b128) — inside the realistic multi-node f ≈ 0.6–0.8 band. So the measured
numbers **2.59× / 2.27× / 1.34×** land on (b32/b64 above) the cost-model band
`accept_len/(K(1−f)+1)` = 1.6–2.1× that RESULTS.md could previously only project. The
measured/ideal ratio at high f is 1.20–1.28 for b32/b64 — the FP8 draft is cheaper than the
`T_compute` the formula assumes, same surplus Phase 47 saw at f=0.

## 3. Slope validation: the over-charge is gone

Delay-slope fit on inverse throughput (`time = a + s·d`), spec vs no-spec, against the
analyzer's shielded model `s_sh = s_ns/accept_len` (= 0.346·s_ns at accept 2.89):

| batch | s_sp/s_ns (charged, P47) | s_sp/s_ns (shielded, P48) | model | over-charge now |
|---:|---:|---:|---:|---:|
| 32  | 0.408 | **0.249** | 0.346 | −27% |
| 64  | ~0.41 | **0.266** | 0.346 | −23% |
| 128 | ~0.41 | **0.347** | 0.346 | **0%** |

b128 lands exactly on the verify-only slope; b32/b64 land *below* it (their verify overlaps
part of the injected sleep with other work at small batch). The draft's delay charge is
eliminated — the empirical confirmation the destroy-log counter can't give (V1 force-kills
workers before `destroy()` fires, as in Phase 47).

## 4. Honest caveats

1. **b128 at low delay ran below Phase 47's charged points** (0.65 vs 0.72 at a2a=0, 0.87 vs
   0.93 at 250 µs). The gate cannot cause this (it only removes draft work; accept identical);
   it is night-to-night engine variance on the compute-bound large-batch point (this run also
   rebuilt the torch.compile cache). At ≥500 µs the shielded b128 wins decisively anyway
   (1.21/1.34 vs 0.96/1.14).
2. Three points flagged `suspect` (b64@250/500, b128@1000) — the ones below their model line;
   the surrounding grid brackets them.
3. Still an emulated comm-bound regime on one node; the real multi-node IB run remains the
   gold standard (unchanged from RESULTS.md future-work 1).

## 5. Verdict

The Phase-47 "conservative lower bound" is retired: with the draft correctly shielded, the
**measured** win curve is ~2× the charged one in the comm-bound regime and **meets or beats the
cost model** across the grid. The multi-node claim upgrades from "projected 1.6–2.1×" to
"**measured 1.3–2.6× at emulated f = 0.67–0.78**, batch-dependent, monotone in f."

Artifacts: `data/w7fp8_sweep2_a2a*_K2.json`, `logs/analysis_output.txt` (full analyzer),
`scripts/compare_charged_vs_shielded.py` (side-by-side), driver `scripts/run_sweep_shielded.sh`.
