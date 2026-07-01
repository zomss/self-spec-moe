# Phase 43 — W7 K-retune: best-K speedup curve vs A2A cost (Qwen3-30B, DP=8/EP=8)

**Model** Qwen3-30B-A3B (128 experts, top-8, 48 layers). **Layout** attention-DP +
EP, tp=1, EP=DP=8. **Fabric** forced-PCIe (`NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0
NCCL_IB_DISABLE=1`). **Spec stack** FP8 full-replica comm-free draft
(`DRAFT_FULL_REPLICA=1 DRAFT_LOCAL_ROUTE=1 DRAFT_FULL_CG=1 COMPILE_CONSISTENT=1`),
greedy. `VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0`. **Method** two-length decode
slope (OUTLEN=160, SHORTLEN=32), CUDA graphs ON, WARMUP=2, ITERS=3, identical
prompts/outlen spec vs no-spec. Harness `w7_fp8_timing.py` (UNCHANGED), driver
`scripts/run_kretune.sh`, analyze `scripts/analyze_kretune.py`. HEAD `ee778e3f5`.
**No vllm/ changes.** batch 64.

Sweep: **K in {2,3}** x `VLLM_SELF_SPEC_EMULATE_A2A_DELAY_US in {0,100,250,500}`
us/collective. **No-spec baselines + K=4 spec reused** from Phase 42
(`../42_comm_sweep/data`, K-independent no-spec — no re-run). speedup = spec/no-spec.

## TL;DR

**K=2 is the best operating point and it CROSSES 1.0x (measured) at a2a=500us/coll**
(779.2 vs 777.8 tok/s = **1.002x**), where K=4 was only 0.782x. K=2 dominates K=3
dominates K=4 at EVERY delay: at a2a=0 the floor rises **0.360 (K4) -> 0.504 (K3) ->
0.550 (K2)**; at 100us **0.567 -> 0.638 -> 0.656**; at 250us **0.574 -> 0.675 ->
0.787**; at 500us **0.782 -> 0.935 -> 1.002**. Fewer draft forwards (cheaper cycle)
beat the marginal accept gain of larger K. accept_len is exactly as predicted and flat
across delays: **K2 ~2.89, K3 ~3.79, K4 ~4.69** (the Phase-41 fix holds; beta~0.94-0.96,
per-position accept 0.95-0.97). **But even the best K (K=2) does NOT cross 1.0x at a
realistic 30-200us A2A:** measured/shielded crossover is ~460-500us; at 200us shielded
K2 is only **0.77x**. **Verdict: K-retune roughly halves the gap and rescues the win in
the comm-bound corner (~500us), but it CONFIRMS the draft-compute bound at realistic
comm — best K still <1.0x at <=200us.**

## 1. Measured speedup table (delay x K, batch 64): spec + no-spec tok/s

speedup = spec tok/s / no-spec tok/s. no-spec + K4 reused from Phase 42 (unchanged).

| a2a us/coll | no-spec tok/s | K2 tok/s | K2 su | K3 tok/s | K3 su | K4 tok/s | K4 su |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0    | 1857.6 | 1021.0 | 0.550 | 936.0 | 0.504 | 668.9 | 0.360 |
| 100  | 1457.2 |  955.9 | 0.656 | 930.4 | 0.638 | 826.9 | 0.567 |
| 250  | 1097.1 |  863.4 | 0.787 | 740.9 | 0.675 | 629.6 | 0.574 |
| 500  |  777.8 |  779.2 | **1.002** | 727.3 | 0.935 | 608.4 | 0.782 |

**accept_len (flat across delays, fix holds):** K2 **2.891**, K3 **3.789**, K4 **4.685**.
Matches the target (K2~2.8, K3~3.7 at beta~0.94). Two K3 points are noisier
(CV 0.09: a2a=0 and a2a=500) but not `suspect`-flagged and not outliers; all K2 points
are clean (CV <=0.021). K3 a2a=0 (936.0) sits oddly close to a2a=100 (930.4) -> its slope
fit is slightly flattened (see Section 3 sensitivity: dropping it moves K3 d* by <2us).

## 2. Best K at each delay — is K2/K3 better than K4?

**K2 is best at every delay; ordering is strictly K2 > K3 > K4.** Margin of K2 over K4:

| a2a us | best K | K2 su | K3 su | K4 su | K2 over K4 |
|---:|---:|---:|---:|---:|---:|
| 0   | K2 | 0.550 | 0.504 | 0.360 | **+0.190 (+53%)** |
| 100 | K2 | 0.656 | 0.638 | 0.567 | +0.089 (+16%) |
| 250 | K2 | 0.787 | 0.675 | 0.574 | +0.213 (+37%) |
| 500 | K2 | 1.002 | 0.935 | 0.782 | +0.220 (+28%) |

The a2a=0 floor is the clearest signal: each extra draft forward is a full 30B-replica
MoE pass whose cost exceeds its marginal accept gain. K2 (2 draft forwards, accept 2.89)
runs the decode cycle cheaper than K3 (accept 3.79) or K4 (accept 4.69), and the higher
throughput floor lifts the whole curve. K3 is a solid second; K4 is dominated everywhere.

## 3. Crossover (speedup = 1.0x) per K — measured AND draft-shielded

**MEASURED (draft over-charged; conservative lower bound):**
- **K2 CROSSES 1.0x at a2a ~500us** (1.002x at 500; linear interp of the 250->500
  segment lands the crossover at **~498us**). K2 is the only K to reach parity <=500us.
- **K3: no crossover <=500us** (max 0.935 at 500; last-two-point extrapolation ~563us).
- **K4: no crossover <=500us** (max 0.782 at 500; extrapolation ~761us; Phase 42's
  full 5-point sweep incl. 1000us gave plateau ~0.82 and no crossover <=1000us).

**DRAFT-SHIELDED (true comm-free multi-node: draft pays 0 A2A; verify slope =
s_ns/accept_len; d=0 compute a_sp unchanged).** Delay-slope fit on inverse-throughput
`time = a + s*d` over the 4-delay grid {0,100,250,500}:

| K | a_sp (d=0) | s_sp/s_ns | accept | shielded s_sh | **shielded d\*** | measured x-over |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 9.87e-4 | 0.408 | 2.89 | 5.17e-7 | **~459us** | ~498us (interp) |
| 3 | 1.07e-3 | 0.462 | 3.79 | 3.95e-7 | **~484us** | ~563us (extrap) |
| 4 | 1.37e-3 | 0.369 | 4.69 | 3.19e-7 | **~705us** | ~761us (extrap) |

(nospec fit: a_ns=5.38e-4, s_ns=1.50e-6/us. `s_sp/s_ns < 1` -> spec is less
delay-sensitive: verify amortized over accept_len + comm-free draft. K3 fit is robust to
its noisy a2a=0 point: dropping it shifts d* from 484 -> 486us.)

Shielded d* is lowest for K2 (~459us) because its d=0 compute floor a_sp is smallest
(fewest draft forwards). Larger K has a lower delay-slope (bigger accept_len denominator)
but a much larger fixed compute cost, and the compute term dominates the crossover here.

## 4. Is the crossover a realistic multi-node A2A cost? (30-200us)

**No.** Even the best K (K2) needs ~460-500us/collective to reach parity. At the
realistic cross-node decode-A2A range (**30-200us**, latency-bound KB-scale messages;
see Phase 42 Section 6) the shielded speedups are:

| a2a us | K2 | K3 | K4 |
|---:|---:|---:|---:|
| 30  | 0.581 | 0.538 | 0.423 |
| 100 | 0.661 | 0.619 | 0.491 |
| 150 | 0.716 | 0.675 | 0.539 |
| 200 | 0.767 | 0.728 | 0.585 |
| 300 | 0.863 | 0.830 | 0.674 |
| 500 | 1.032 | 1.014 | 0.842 |

**At <=200us, best-K (K2) shielded speedup is 0.58-0.77x** — still sub-parity. Parity
only arrives at ~500us (a slow/large-message regime), the same extreme-comm corner
Phase 42 identified, now shifted DOWN from ~625us (K4) to ~460us (K2) by dropping K.

## 5. Verdict — does K-retune rescue the win at realistic comm?

**Partially, in the comm-bound corner only; the draft-compute bound is confirmed at
realistic comm.**

- K-retune is a real, large lever: **K2 nearly halves the parity gap vs K4** (a2a=0 floor
  0.360 -> 0.550; shielded crossover 705us -> 459us) and **crosses 1.0x measured at
  500us**, which K4 never did within 1000us. K=4 was the wrong operating point (Phase 42's
  hypothesis 1, now confirmed quantitatively): **K should be set to 2**, not 4.
- **But best-K still does NOT win at realistic multi-node A2A.** At the plausible
  30-200us cross-node decode collective, even shielded K2 is 0.58-0.77x. The crossover
  (~460-500us) sits at the HIGH end / above the realistic range. The dominant limiter is
  therefore the DRAFT COMPUTE, not K and not the comm-emulation caveat: K2 x 30B-replica
  forwards leave spec sub-parity at any realistic A2A.
- **Implication (unchanged direction, sharpened):** the remaining wall is the 30B
  full-replica draft's per-step compute (~K x target MoE FLOPs). K-retune moved the
  crossover from ~700us to ~460us; closing the last ~460 -> ~150us needs a structurally
  CHEAPER draft (fewer experts/layers or a distilled head), i.e. Phase-34/41's **W2b**.
  World A needs a cheap draft, not merely a comm-bound fabric — K=2 gets partway there.

## 6. Emulation fairness / draft-shielding caveat (carried over from Phase 42)

Confirmed identical to Phase 42: the full-replica draft's MoE takes the `naive_dp_ep`
fallback (log: `MoEPrepareAndFinalizeNaiveDPEPModular` + "Detected DP deployment with no
--enable-expert-parallel. Falling back to AllGather+ReduceScatter"), so the draft issues
ZERO real collectives (local-route) but STILL reaches `_emulate_exposed_a2a_delay()` (the
`torch.cuda._sleep` is not gated by `self_spec_local_route_enabled()`). The draft thus
eats the injected sleep -> **the MEASURED speedups OVER-charge spec (conservative lower
bound)**; a true multi-node comm-free draft pays 0 fabric latency. We report both the
measured curve and the draft-shielded projection (draft pays 0 A2A; verify slope =
s_ns/accept_len). No-spec genuinely pays the delay (its tok/s falls 1857.6 -> 777.8 from
a2a 0 -> 500, b64), so the emulation is not the "only-spec-pays" failure mode; the
residual bias is conservative. Fixing it exactly is a vllm/ change, out of scope.

## Files
- Driver `scripts/run_kretune.sh` (spec-only; no-spec + K4 reused from
  `../42_comm_sweep/data/`); harness (UNCHANGED)
  `research/34_worldA_system/scripts/w7_fp8_timing.py`.
- Analysis `scripts/analyze_kretune.py` (measured table, best-K, delay-slope +
  draft-shielded projection + crossover).
- Data `data/w7fp8_kr_a2a{0,100,250,500}_fp8_spec_*_K{2,3}.json`;
  logs `logs/kr_a2a*_*.log`, `logs/driver.log`.
