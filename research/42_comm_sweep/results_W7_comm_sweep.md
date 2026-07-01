# Phase 42 — W7 comm sweep: speedup vs emulated A2A cost (Qwen3-30B, DP=8/EP=8)

**Model** Qwen3-30B-A3B (128 experts, top-8, 48 layers). **Layout** attention-DP +
EP, tp=1, EP=DP=8. **Fabric** forced-PCIe (`NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0
NCCL_IB_DISABLE=1`). **Spec stack** FP8 full-replica comm-free draft
(`DRAFT_FULL_REPLICA=1 DRAFT_LOCAL_ROUTE=1 DRAFT_FULL_CG=1 COMPILE_CONSISTENT=1`),
K=4, greedy. `VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0`. **Method** two-length
decode slope (OUTLEN=160, SHORTLEN=32), CUDA graphs ON, WARMUP=2, ITERS=3, identical
prompts/outlen spec vs no-spec. Harness `w7_fp8_timing.py` (UNCHANGED), driver
`scripts/run_sweep.sh`, analyze `scripts/analyze_sweep.py`. HEAD `dd842e17e`.

Sweep: `VLLM_SELF_SPEC_EMULATE_A2A_DELAY_US in {0,100,250,500,1000}` us/collective x
batch {32,64,128}. speedup = spec tok/s / no-spec tok/s.

## TL;DR

At **K=4** the FP8 full-replica comm-free draft **never crosses 1.0x within 1000us**
as measured — speedup rises with A2A cost but **plateaus at ~0.82x** (b64). The rise
is monotonic (0.36 -> 0.57 -> 0.57 -> 0.78 -> 0.82) and confirms the comm-bound
mechanism (no-spec decode falls 3.8x from 0->1000us while spec falls only ~1.7x),
but on this box the K=4 30B-replica draft's compute keeps spec below parity even in
a deeply comm-bound regime. Accept is fully recovered and flat (~4.68-4.76 across all
delays/batches) — the Phase-41 fix holds. The **draft-shielded projection** (true
comm-free multi-node, where the draft pays ZERO fabric latency and only the verify
pays it) crosses **1.0x at ~580-630us/collective (b32/b64), ~880us (b128)** and
reaches **1.28-1.35x at 1000us**. Whether World A "wins where designed" hinges on
whether ~600-900us/collective is a realistic cross-node MoE all-to-all — it is at the
HIGH end of the plausible range (see Section 6): realistic tens-to-few-hundred-us
fabrics land World A around **0.7-0.9x** even shielded. **K=4 is the wrong operating
point** (K=3 was ~0.45/0.62; K=4's extra draft forward costs more than its marginal
accept gain) -> the implication is W2b: a cheaper draft, not merely a comm-bound
fabric.

## 1. Measured speedup table (delay x batch): spec + no-spec tok/s

speedup = spec tok/s / no-spec tok/s. accept_len stable ~4.68-4.76 (fix holds).

| a2a us/coll | batch | no-spec tok/s | spec tok/s | accept_len | **speedup** |
|---:|---:|---:|---:|---:|---:|
| 0    | 32  | 1192.4 | 382.4  | 4.72 | 0.321 |
| 0    | 64  | 1857.6 | 668.9  | 4.68 | 0.360 |
| 0    | 128 | 2698.1 | 964.0  | 4.67 | 0.357 |
| 100  | 32  | 873.7  | 463.4  | 4.73 | 0.530 |
| 100  | 64  | 1457.2 | 826.9  | 4.69 | **0.567** |
| 100  | 128 | 2235.5 | 1230.9 | 4.67 | 0.551 |
| 250  | 32  | 627.9  | 351.8  | 4.74 | 0.560 |
| 250  | 64  | 1097.1 | 629.6  | 4.69 | 0.574 |
| 250  | 128 | 1779.9 | 980.4  | 4.67 | 0.551 |
| 500  | 32  | 428.0  | 346.4  | 4.76 | 0.809 |
| 500  | 64  | 777.8  | 608.4  | 4.69 | 0.782 |
| 500  | 128 | 1337.6 | 958.9  | 4.67 | 0.717 |
| 1000 | 32  | 259.3  | 215.1  | 4.70 | 0.830 |
| 1000 | 64  | 488.3  | 401.5  | 4.69 | **0.822** |
| 1000 | 128 | 892.2  | 700.2  | 4.69 | 0.785 |

**Sanity vs Phase 41 (K=3):** no-spec b64 a2a=0 = 1857.6 (P41: 1876.3, <1%),
a2a=100 = 1457.2 (P41: 1467.9, <1%), b128 a2a=0 = 2698.1 (P41: 2688.7). Baselines
reproduce. At K=4 the b64 speedup is 0.360 (a2a=0) / 0.567 (a2a=100) vs P41's K=3
0.45 / 0.62 -> **K=4 is worse** (below): the extra draft forward costs more than its
marginal accept gain (accept_len 4.68 at K=4 vs 3.78 at K=3, but 4 x 30B-draft
forwards vs 3).

## 2. Crossover (speedup = 1.0x) per batch

**MEASURED (draft over-charged; conservative): NO crossover <= 1000us for any batch.**
Max measured speedup 0.83/0.82/0.79 (b32/b64/b128). The curve is concave/saturating:
100->250us barely moves (0.567->0.574 at b64) then re-steepens 250->1000us. A
last-two-point linear extrapolation gives ~2600-5200us but is unreliable given the
saturation; the honest statement is **no crossover in any realistic delay for the
measured (draft-charged) config at K=4.**

**DRAFT-SHIELDED (true comm-free multi-node; draft pays 0 A2A, only verify pays,
amortized over accept_len):**

| batch | shielded crossover (speedup=1.0) | shielded speedup @1000us |
|---:|---:|---:|
| 32  | **~580us** | 1.354 |
| 64  | **~625us** | 1.283 |
| 128 | **~880us** | 1.069 |

## 4. Delay-slope fit (why spec is less delay-sensitive) + shielded projection

Fit decode-time-per-fixed-token-count vs delay `d`: `time = a + s*d` (least squares
on 1/tok_s). `s` = per-collective delay coefficient; `s_sp/s_ns` = spec's delay
sensitivity relative to no-spec.

| batch | s_ns (per us) | s_sp (per us) | **s_sp/s_ns** | accept_len |
|---:|---:|---:|---:|---:|
| 32  | 3.01e-6 | 2.22e-6 | 0.735 | 4.75 |
| 64  | 1.51e-6 | 1.12e-6 | 0.740 | 4.69 |
| 128 | 7.49e-7 | 4.85e-7 | 0.648 | 4.67 |

`s_sp/s_ns < 1` confirms spec pays LESS A2A per output token than no-spec (verify
amortized over accept_len + comm-free draft). Shielded slope = `s_ns/accept_len`
(verify-only, amortized). Shielded speedup:

| a2a us | b32 | b64 | b128 |
|---:|---:|---:|---:|
| 0    | 0.379 | 0.421 | 0.419 |
| 100  | 0.501 | 0.526 | 0.495 |
| 250  | 0.672 | 0.675 | 0.603 |
| 300  | 0.726 | 0.722 | 0.638 |
| 400  | 0.829 | 0.813 | 0.705 |
| 500  | 0.928 | 0.900 | 0.771 |
| 1000 | 1.354 | 1.283 | 1.069 |

## 5. Accept + losslessness

accept_len is **flat 4.67-4.76 across every delay and batch** (the emulated delay is
pure wall-clock; it does not touch numerics). This reconfirms the Phase-41 fix: no
DP>=4 collapse (was ~1.0 pre-fix at DP=8). accept_len 4.69 at K=4 vs 3.78 at K=3
(FP8) is the expected K-length effect. Greedy, seed=0, ignore_eos.

## 6. Interpretation — is the crossover a realistic multi-node A2A cost?

**Message-size regime.** Qwen3-30B-A3B: hidden=2048, bf16. The decode-step MoE
dispatch all-gathers hidden_states across the 8 DP ranks and combine reduce-scatters.
Per token the payload is `hidden x 2B = 4 KB`; per rank per step (B/dp tokens) it is
**~16-64 KB for batch 32-128** (tens of KB total across the ring). At 400 Gb/s the
bandwidth term for 32 KB is ~0.7us -> **decode A2A is LATENCY-bound, not
bandwidth-bound**. A realistic cross-node NCCL all-gather / reduce-scatter of a
KB-scale message is dominated by per-hop + launch latency: **~30-200us/collective**
on a good IB/RoCE fabric, higher (few hundred us) on an oversubscribed or
higher-hop-count topology or with larger batches / prefill-mixed steps.

**Verdict on the crossover.**
- **Measured (draft charged the emulated delay): World A does NOT reach parity at
  K=4 for any batch within 1000us/collective** (max 0.79-0.83x). Even in a strongly
  comm-bound emulated regime the K=4 30B-replica draft's compute keeps it below 1.0x.
- **Draft-shielded (true comm-free multi-node, the semantics World A is designed
  for): crossover ~580-880us/collective.** That is at the **HIGH end / above** the
  realistic 30-200us cross-node decode-A2A range. At a realistic **300-500us**
  (already a slow/large-message regime) the shielded speedup is only **0.64-0.93x**
  (Section 4). World A reaches >1.0x only if the per-collective A2A is ~600us+, which
  implies a slow fabric OR large batched/long-context A2A messages, not the typical
  tens-to-low-hundreds-us decode collective.

**So World A "wins where designed" only in an EXTREME comm-bound corner** (>=600us
A2A), not at typical multi-node decode A2A costs. The dominant limiter is the DRAFT
COMPUTE, not the emulation caveat: even fully shielded, at a realistic 300-500us the
draft's K=4 x 30B forwards leave spec at 0.64-0.93x.

## 7. Implication (no crossover <= 1000us measured; shielded crossover too high)

The comm-free full-replica draft is **too expensive** to win at realistic multi-node
A2A costs, even though it is correct, lossless, and accept-recovered (~4.7). Two
concrete levers, in priority order:

1. **K is mis-set. Drop to K<=3 (or tune K).** K=4 is dominated by K=3 here (K=3 was
   0.45/0.62 at a2a 0/100 in Phase 41; K=4 is 0.36/0.57): the 4th draft forward costs
   a full 30B-replica pass for a marginal accept gain (4.68 vs 3.78 accept_len). At
   the crossover-relevant delays a smaller K raises the a2a=0 floor and thus the whole
   curve.
2. **W2b: a CHEAPER draft.** The 30B full-replica draft (even FP8) costs ~K x the
   target MoE FLOPs per step; that is the wall that caps the shielded curve at
   0.64-0.93x for realistic A2A. A smaller/pruned/low-layer draft (fewer experts,
   fewer layers, or a distilled head) would cut the d=0 compute floor `a_sp` and move
   the crossover well into the realistic 30-200us range. This is the phase-34/41
   structural conclusion, now quantified: **World A needs a cheap draft, not merely a
   comm-bound fabric.**

## Files
- Driver `scripts/run_sweep.sh`; harness (UNCHANGED)
  `research/34_worldA_system/scripts/w7_fp8_timing.py`.
- Analysis `scripts/analyze_sweep.py` (measured table + crossover),
  `scripts/crossover_analysis.py` (delay-slope fit + draft-shielded projection),
  `scripts/probe_a2a_counts.py` (fairness probe).
- Data `data/w7fp8_sweep_a2a{0,100,250,500,1000}_{nospec,fp8_spec}_*.json`;
  logs `logs/sweep_a2a*_*.log`, `logs/sweep_driver.log`.

## 3. Emulation fairness (verified BEFORE trusting numbers)

The emulated delay is injected in `AgRsAll2AllManager._emulate_exposed_a2a_delay()`
(all2all.py:88) as `torch.cuda._sleep(...)` on the current stream (captured into the
CUDA graph so it replays every decode step). It is called from `dispatch`,
`dispatch_router_logits`, and `combine` for ALL branches. Findings:

- **No-spec pays it (REQUIRED, confirmed).** The plain full-EP decode constructs the
  real EP path; every MoE collective is a real `all_gatherv`/`reduce_scatterv` and
  also runs the injected sleep. Load log: `Self-spec A2A delay: 100.0 us/collective
  on GPU stream (1979.7 cycles/us)`. No-spec b64 drops 1857.6 (a2a=0) -> 1457.2
  (a2a=100) -> ... , i.e. it genuinely pays the delay on every collective per token.

- **The comm-free draft's REAL collectives = 0 (confirmed structurally).** The
  full-replica draft is built `use_ep=False, ep_size=1, tp_size=1` (config.py:1211
  branch); it issues NO real `all_gatherv`/`reduce_scatterv` (real=0). Verified in
  Phase 41 (bit-identical to plain MoE, rel-err 0.0).

- **CAVEAT — the draft still pays the *emulated* sleep (over-charge).** Because the
  draft runs with `dp_size=8, use_ep=False`, `maybe_make_prepare_finalize`
  (all2all_utils.py:144) takes the "Detected DP deployment with no
  --enable-expert-parallel. Falling back to AllGather+ReduceScatter" path and the
  draft's MoE builds `MoEPrepareAndFinalizeNaiveDPEPModular` (log line at fp8.py:594).
  Its forward calls `get_ep_group().dispatch/combine`, where the local-route branch
  skips the REAL collective (real stays 0) but STILL reaches
  `_emulate_exposed_a2a_delay()` -> the `torch.cuda._sleep` is NOT gated by
  `self_spec_local_route_enabled()`. So the draft eats the injected sleep on every
  MoE layer, K times per step, even though it does zero cross-rank comm.

  **Direction of the bias:** this OVER-charges the spec path (a true comm-free
  multi-node draft would pay ZERO fabric latency). The measured speedups are
  therefore a **conservative lower bound** on World A. Fixing it exactly would
  require gating the sleep on `self_spec_local_route_enabled()` in all2all.py (a
  vllm/ change, out of scope). We report the measured (over-charged) curve AND an
  analytic draft-shielded projection (Section 4).

  **Numerically the over-charge is moderate, empirically.** The measured delay-slope
  ratio `s_sp/s_ns` (how fast decode-time-per-token grows with the emulated delay,
  spec vs no-spec, full-dataset fit) is **0.65-0.74** (Section 4). If the draft paid
  the full delay on all K x C collectives, spec would be MORE delay-sensitive than
  no-spec (ratio > 1). Instead spec is ~1.4x LESS delay-sensitive: no-spec pays the
  A2A on every token; spec's residual delay cost is the VERIFY (once/step, amortized
  over accept_len ~= 4.68) PLUS a non-trivial residual from the draft's own
  local-route dispatch/combine still running the injected sleep. Removing that
  residual (draft-shielded, Section 4) shifts the crossover DOWN by ~35-40%.

  This is NOT the failure mode the task warned about ("emulation only affects spec,
  not no-spec"): no-spec definitely pays (its decode tok/s falls 3.8x from a2a=0 to
  a2a=1000, b64). The residual bias is the opposite and conservative.
