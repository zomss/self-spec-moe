# Phase 73 results — the b1 draft forward is FIXED-OVERHEAD-dominated (amortizes)

Batch-scaling of the **PIECEWISE** fp8-full-replica shared-KV W512 K4 draft,
Qwen3-30B-A3B single-node DP8/EP8 on h106, SHORT 2k context (so per-rank batch
scales free of the 16k KV-pool cap). SelfSpecProfiler coarse regions
(`draft_forward` = one draft model forward inside the chain; `verify` = target
verify forward), sample-weighted mean across all 8 ranks, 40-sample warmup.

## Batch-scaling table (per-rank decode batch = W7_BATCHES)

**Per-rank mapping (MEASURED, corrects the mission premise).** This is a
*replicated-batch* DP harness: each of the 8 DP ranks independently runs the
full `W7_BATCHES`-prompt batch (they lockstep only for the EP all-to-all).
Confirmed three ways at steady state: `W7_BATCHES=8` -> draft `bs=8`
(kv-window call=501), GPU KV usage 6.9% = 8 reqs x 2051 tok / 239,120-tok pool,
"Running: 8 reqs"/engine; `W7_BATCHES=16` -> `bs=16`, 14.1%; `b64` -> 49%. So
the label **b\<N\> == per-rank decode batch N** (NOT N/8). (P72's "b8->b1/rank"
was the FULL-CG path, which the harness README labels b1/rank; the PIECEWISE
draft measured here batches all concurrent requests.)

| per-rank b | draft_fwd ms | **df/tok** | df_first ms | verify ms | vf/tok | chain ms | accept | tok/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1  | **6.87** | 6.87 | 10.13 | 19.11 | 19.11 | 37.39 | 4.835 |   64.4 |
| 2  | 7.60 | 3.80 | 11.66 | 19.87 |  9.93 | 40.95 | 4.237 |  107.5 |
| 4  | 8.21 | 2.05 | 11.52 | 21.14 |  5.28 | 42.64 | 4.363 |  195.5 |
| 8  | 9.74 | 1.22 | 12.59 | 23.49 |  2.94 | 51.20 | 4.525 |  350.4 |
| 16 | 11.42 | 0.71 | 14.09 | 27.33 |  1.71 | 53.86 | 4.615 |  622.7 |
| 32 | 12.02 | 0.38 | 15.57 | 35.52 |  1.11 | 57.19 | 4.543 | 1031.9 |
| 64 | 13.33 | 0.21 | 17.43 | 49.04 |  0.77 | 63.01 | 4.538 | 1607.9 |

accept_len ~4.2-4.8 across the sweep (sanity OK). The draft forward grows from
**6.87 ms at b1 to 13.33 ms at b64 — only 1.94x for a 64x increase in tokens.**

## 1. Fit  draft_forward(b) = F_fixed + b*m

The curve is **concave** (marginal per-token work falls as the batch fills the
GPU), so one straight line is an approximation; segmented fits bracket it:

| fit | F_fixed (ms) | m (ms/token) | note |
|---|---:|---:|---|
| global lstsq (b1..64) | 8.17 | 0.095 | R^2=0.78 (concave) |
| **low-batch (b<=8)** | **6.64** | **0.392** | intercept ~= the b1 floor |
| high-batch (b>=16) | 10.76 | 0.040 | asymptotic marginal |

**F_fixed ~= 6.6-6.9 ms is 97-100% of the b1 forward.** The batch-independent
overhead dominates; the marginal per-token draft work is m ~= 0.39 ms/tok at
low batch, falling to ~0.04 ms/tok once the GPU is filled (avg 0.095 over the
whole range). For comparison the mission's illustrative case
(F~=10.6, m~=0.6, 96% fixed) is the same shape one context-band up.

## 2. Per-token draft cost vs batch (the amortization curve)

`df/tok` column above: **6.87 -> 3.80 -> 2.05 -> 1.22 -> 0.71 -> 0.38 -> 0.21
ms/token** for b = 1,2,4,8,16,32,64. The fixed ~6.9 ms spreads over the batch:
**33x cheaper per token from b1 to b64.** `verify` amortizes the same way but
shallower (vf/tok 19.1 -> 0.77, m=0.474 ms/tok — verify does real K+1-token
work per request, so a larger irreducible marginal than the draft's 0.04-0.39).

## 3. GEMM-only floor (the hard fusion floor)

P72 `decompose_draft_forward.txt`, the essential data-dependent GEMMs of one
b1 draft forward (48 layers): fused_moe 2.249 + SDPA bmm 1.912 + cutlass-fp8
qkv/o 0.944 = **5.11 ms**. No fusion can beat this sequential-GEMM sum at b1.
The measured 2k b1 forward (6.87 ms) sits at **1.34x the floor** — i.e. only
~1.76 ms of reducible glue above the GEMM floor at 2k. (P72's 16k b1 forward
was ~11 ms = ~5 ms GEMM + ~6 ms glue: the tiny memory-bound plumbing kernels
inflate ~3x under 16k HBM pressure, so the reducible glue is context-dependent,
~1.8 ms at 2k vs ~6 ms at 16k.) F_fixed(6.6) / GEMM-floor(5.11) = 1.30x.

## 4. Verdict + accessibility

**Verdict: FIXED-OVERHEAD-DOMINATED. The user's hypothesis is CONFIRMED.** The
b1 draft forward is ~97-100% batch-independent overhead (F_fixed ~= 6.9 ms) with
a tiny per-token marginal (m ~= 0.04-0.39 ms/tok). It amortizes hard: a 64x
batch costs only 1.94x the forward, and per-token draft cost falls 33x (6.87 ->
0.21 ms/tok). So the ~6 ms of glue P72 flagged is **reducible overhead that
amortizes with tokens/rank**, not a per-token floor. Two independent levers:
(a) batch it (amortize F_fixed over more tokens/rank); (b) fuse the glue toward
the 5.11 ms GEMM floor (removes ~1.8 ms at 2k, ~6 ms at 16k of tiny kernels).

**Accessibility at 16k 2-node MoE-EP: LARGELY NOT REACHABLE.** The amortization
needs high per-rank batch, but the 16k KV pool caps it. Phase 66 (same harness,
same fp8-replica shared-KV arm): pool = 243,632 tok/rank, demand 16,313 tok/req,
so **max clean per-rank batch ~= 12** (80% pool); b13 (87%) already shows 88
preemption waves and b32 livelocks. At the b12 pool ceiling the draft is
~10.6 ms, per-token ~0.88 ms/tok — **~7.8x amortized vs b1, but still 4x above
the b64 floor (0.21 ms/tok)**, which would need per-rank 64 = 1.04M tok/rank =
4.3x the available pool. Worse, real DP-attention serving shards the batch
(per-rank = B_total / DP_width): reaching per-rank 8-12 needs ~130-190
concurrent requests on DP16, i.e. near pool saturation / the preemption knee;
at ordinary load the wide DP pins per-rank to ~1-2, where the draft runs at
~7-8 ms for 1-2 tokens (~3.4-6.9 ms/token) — essentially **unamortized**.

**Bottom line:** the b1 draft forward is reducible fixed overhead in principle
(it amortizes ~33x by b64 and fuses toward a 5.11 ms floor), but at 16k
MoE-EP serving the KV pool + wide attention-DP keep per-rank batch at ~1-12,
capping the achievable amortization at ~8x and pinning it near ~1-2 (no
amortization) at normal load. Fixed-overhead-dominated, yes; but the batch that
would make the draft cheap per token is not accessible in the target regime.

## 16k cross-check (ctx-invariance of the PIECEWISE draft)

One 16k point at b8 (per-rank 8, confirmed: steady draft `bs=8`; measured 16k
pool = 239,104 tok/rank, max concurrency 11.68x for 20,480-tok requests, i.e.
~12 reqs/rank of 16k context fit — matching the Phase-66 ceiling):

| region @ b8 | 2k | 16k | delta |
|---|---:|---:|---:|
| **draft_forward** | 9.74 | 9.98 | **+2.4%** (ctx-invariant) |
| draft_forward_first | 12.59 | 14.52 | +15% (step-0 rebuilds window from full ctx) |
| verify | 23.49 | 30.62 | +30% (verify attends the full 16k KV) |
| accept_len | 4.525 | 4.624 | -- |

**The windowed draft forward is context-invariant (+2.4%)** — the cap544 KV
window makes it independent of the 16k context, exactly as premised, so the 2k
amortization curve transfers to 16k. Only `verify` (full-context attention) and
the step-0 window-build inflate with context; the per-chain-step `draft_forward`
that the amortization analysis is built on does not. So the 16k-accessibility
conclusion above uses the measured 2k curve directly.

## Repro

`scripts/run_sweep.sh` (runs ON h106; strict foreign-GPU guard; one engine per
point). Sweep: `POINTS="2048:1 2048:2 2048:4 2048:8 2048:16 2048:32 2048:64
16384:8"` (the 2k sweep was split across two invocations). `scripts/analyze.py`
builds this table + fits. Data: `data/prof_ctx<CTX>_b<B>/`, `data/w72n_*.json`.
