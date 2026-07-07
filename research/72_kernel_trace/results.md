# Phase 72 results — kernel trace CONFIRMS the latency floor; MoE is small & sparse

Kernel-level kineto trace of rank 0, Phase-70 full-CG draft arm (DP8/EP8, 16k,
W512 cap544, fp8 full-replica comm-free draft, K4 b8 -> b1/rank), **FULL-CG
REPLAY confirmed** on all 8 ranks (b1 descriptor `num_tokens=1, num_reqs=1,
uniform=True`). One draft-chain forward = one b1 pass down all 48 layers.

Raw trace: `data/trace_fullcg_w512k4_b8/dp0_pp0_tp0_dcp0_ep0_rank0.<id>.pt.trace.json.gz`
(15 MB, gitignored). Per-forward table: `data/per_forward_timeline.txt`.
Full decomposition: `data/decompose_draft_forward.txt`.

## 1. Latency-floor signature (steady-state regular chain step)

| metric | value |
|---|---|
| wall (GPU timeline, first->last kernel) | **11.35 ms** |
| GPU-active (union of busy intervals) | **9.77 ms (86.1% of wall)** |
| idle / inter-kernel gap fraction | **13.9% (1.58 ms)** |
| # kernels in the forward | **2403** (~50 kernels/layer x 48 layers) |
| median kernel duration | **1.89 us** |
| mean kernel duration | 4.06 us |
| median inter-kernel gap | **0.42 us** |
| mean inter-kernel gap | 0.66 us |

The forward is a serial chain of **~2400 tiny kernels** (median 1.9 us — near the
minimum kernel granularity), packed back-to-back (median gap 0.42 us) at **86%
GPU-active**. It is **latency / kernel-granularity bound**: the floor is the
sheer *count* of tiny per-layer kernels, each doing trivial b1 work, not weight
bandwidth and not a fat GEMM.

### Wall distribution (`data/per_forward_timeline.txt`)

Draft-forward wall is **bimodal + a startup transient**, and its range exactly
matches P70's independently-measured `draft_forward` range (P70: min 10.6, max
30.6 ms; here: min 11.1, max 31.7 ms):

- **Regular chain steps 1-3:** median **11.23 ms** (n=58; 11.13-11.50, very tight).
- **Step-0 of each K=4 chain:** median **15.08 ms** (n=18; the extra ~4 ms is a
  larger first-step boundary, consistent with P70's `draft_forward_first` 14.3 ms).
- **Startup / prefill-phase transient (first ~2 s):** up to **31 ms** — the
  *same* ~2500 kernels running ~3x slower. This is **95% GPU-busy (only ~5%
  idle)**, and the slowdown is **non-uniform** (the tiny memory-bound
  elementwise/copy kernels inflate; the compute-bound cutlass GEMMs do not) =>
  HBM contention during the 16k prefill phase, **not** a clock effect and **not**
  idle gaps. Cross-check: the steady target-verify forward is 19.2 ms here vs
  P70's synced 40.8 ms — the *same* ~2x factor, so both P70 numbers were read in
  the inflated regime + carry the region's two `cuda.synchronize()`.

**Reconciliation of "the 25 ms":** the steady-state GPU-wall of a draft forward
is **~11 ms/step (step-0 ~15 ms)**, not 25 ms. P70's `draft_forward` mean (25.4
ms) is the mean over a window weighted by the startup transient plus the
synced-region overhead; it is not the steady floor. Because the steady 11 ms
forward is measured at the *end* of the trace (highest profiler overhead) yet is
the *fastest*, 11 ms is an upper bound on the unperturbed rate. This **refines**
P71's absolute number downward but leaves its mechanism intact — and note 11 ms
sits right at P71's read-everything ceiling, so the steady forward is squarely a
compute/latency floor.

## 2. Category split of ONE draft forward (steady regular step, active-basis)

| category | ms | % GPU-active | % wall | #kernels |
|---|---|---|---|---|
| **MoE** (fused_moe GEMM + router + silu + moe-rms) | 2.63 | **27.0%** | 23.2% | 240 |
| **attention** (scratchpad SDPA: 2 bmm + gather + softmax) | 2.21 | **22.7%** | 19.5% | 288 |
| dense linear (fp8 cutlass qkv/o proj) | 0.94 | 9.7% | 8.3% | 144 |
| fp8 quant (per-token scale + cast + fused quant-norm) | 0.69 | 7.1% | 6.1% | 336 |
| KV write (`reshape_and_cache_flash`) | 0.10 | 1.0% | 0.9% | 48 |
| norm/rope (see note) | ~0 | ~0% | — | 1 |
| **misc / per-layer plumbing** | 3.18 | **32.5%** | 28.0% | 1346 |
| TOTAL (GPU-active) | 9.77 | 100% | 86.1% | 2403 |

Note: RMSNorm+residual and RoPE are **fused** into Triton kernels whose names
carry `moe_forward` (counted in MoE) or `to_copy_abs_clamp_..._fused_add_rms_norm`
(counted in fp8-quant), so the explicit `norm/rope` row is ~0 — norm work is
distributed, not absent. `misc` is ~1350 generic elementwise/reduce/copy/memset
kernels: this includes the SDPA softmax reduces + attention mask (really
attention) and the MoE expert-weight reductions (really MoE), plus residual adds
and dtype casts. On a fully-attributed basis attention and MoE each land
~25-35%; the split is roughly regime-invariant (transient 30 ms forward: MoE
24.7% / attn 25.1%; step-0: MoE 22.9% / attn 19.3%).

**Headline — MoE vs attention:** MoE **~27%** (2.6 ms) vs scratchpad attention
**~23%** (2.2 ms). Nearly equal; **neither dominates.** The three real GEMMs
(`fused_moe` 2.25 ms, `bmm` SDPA 1.91 ms, `cutlass fp8` qkv/o 0.94 ms) sum to
~5 ms; the other ~5 ms is ~2000 tiny plumbing kernels spread across 48 layers.

## 3. Sanity vs the arithmetic — MoE is SPARSE top-8, not dense

The MoE expert grouped-GEMM (`fused_moe_kernel`, 96 launches = up-gate + down x
48 layers) totals **2.25 ms** for the whole b1 forward. A **dense** 128-expert
MoE would read ~29 GB fp8 => **~10 ms of MoE GEMM alone**, which by itself would
exceed the entire 9.77 ms GPU-active forward. Since fused_moe is only 2.25 ms,
the draft MoE is unambiguously **SPARSE top-8** (~1.8 GB / ~0.6-0.9 ms of pure
bytes + small-M launch/occupancy overhead). The router kernel confirms it:
`topkGating<8, 128, ...>` selects top-**8** of **128** experts.
=> **CONFIRMS P71, refutes any dense-MoE reading.**

## 4. Scratchpad SDPA — notable but not dominant

The draft attention is PyTorch's **math-backend SDPA** (the masked, `enable_gqa`
`F.scaled_dot_product_attention` over the fixed cap544 scratchpad): 2
`bmm_kernel`/layer (QK^T + attn·V) = **1.91 ms** + `vectorized_gather` K/V gather
= 0.18 ms + softmax/mask in misc. Total ~2.2 ms (~20% of the forward) —
**comparable to MoE**, a real cost, and the unfused 2-bmm+softmax path is less
efficient than a fused FA kernel would be. But it does **not** dominate the 25/11
ms and is not the confound that would overturn the picture.

## Verdict

**CONFIRMS the latency floor.** One b1 draft forward = a deep serial chain of
**~2400 tiny kernels** (~50/layer x 48 layers, median 1.9 us) at **86% GPU-active
(only 14% inter-kernel idle)**. It is **latency / kernel-granularity bound**, not
weight-bandwidth bound (MoE GEMM 2.25 ms) and not a fat dense MoE (sparse top-8;
a dense MoE alone would be ~10 ms). **MoE ~27% ≈ attention ~23%; neither
dominates; the bulk is per-layer plumbing.**

Two refinements to P71: (a) the steady-state GPU-wall is **~11 ms/step (step-0
~15 ms)**, not 25 ms — the 25 ms was region-sync overhead + a startup/prefill
transient (identical kernels ~3x slower under HBM contention), and P70's range
(10.6-30.6) matches this trace exactly; (b) the floor is dominated by the *number*
of tiny serial kernels (86% active), **not** by large GPU-idle gaps. The
structural levers stand: shallower/smaller draft (fewer serialized kernels),
fewer experts/token, or more tokens/rank — full-CG coverage is not the lever.
