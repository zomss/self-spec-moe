# Phase 44 — W7 recon: why comm-free self-spec is 0.55x, not the model's ~1.1x

**Model** Qwen3-30B-A3B (128 experts, top-8, 48 layers). **Layout** attention-DP +
EP, tp=1, EP=DP=8. **Fabric** forced-PCIe (`NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0
NCCL_IB_DISABLE=1`), a2a delay **0** (native forced-PCIe). **Spec stack** FP8
full-replica comm-free draft (`DRAFT_FULL_REPLICA=1 DRAFT_LOCAL_ROUTE=1
DRAFT_FULL_CG=1 COMPILE_CONSISTENT=1`), greedy, **K=2, batch 64**, CUDA graphs ON.
`VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0`. HEAD `11f4777fe`. **No vllm/ changes.**
Timing via the env-gated profiler (`vllm/v1/spec_decode/self_spec_profiler.py`,
regions `draft_forward_first / draft_forward / draft_chain / verify`), rank-0 steady
means (WARMUP=60). Harness `scripts/recon_microbench.py`, driver `scripts/run_recon.sh`,
analysis `scripts/analyze_recon.py`. Data in `data/`, logs in `logs/`.

## TL;DR

The cost model predicts **1.166x** (`E_tok·T_verify(64) / [K·T_compute(64) +
T_verify(192)] = 2.92·25.9 / 64.7`); the measured cycle gives **0.550x** (reproduces
the K-retune 0.55x exactly). The cost model's **64.7 ms cycle is 2.83x too small:
the real cycle is 182.9 ms.** The 118.1 ms gap splits **draft-GPU +53.5 ms (45%),
verify +33.2 ms (28%), unmodeled CPU orchestration +31.4 ms (27%)** — and it sums
*exactly* to the gap. **The dominant error is the draft: the real accept-achieving
draft forward is 67.5 ms/step = 4.15x the T_compute(64)=16.3 ms the model assumed.**
Root cause: on this non-MLA MoE the draft chain runs its **attention eagerly**
("the captured decode graph is not replay-safe for the draft's growing sequence")
and its GEMM/MoE through the batch-invariant (COMPILE_CONSISTENT) persistent-matmul
path — neither is in the captured graph T_compute was measured on. T_compute (16.3 ms)
matches only the **step-0 / whole-model captured forward** (measured 17-18 ms), NOT
the in-chain K-step forwards. Even making the draft as cheap as T_compute only lifts
0.55x -> 0.78x; all three errors must be fixed to reach parity.

## 1. Cycle decomposition (ms, rank-0 steady means, K=2 b64 a2a=0)

Authoritative cycle = the K-retune prefill-cancelled decode point: **1021 tok/s ->
2.918·64 / 1021 = 182.9 ms/cycle** (the 0.55x operating point). accept_len measured
here **2.918** (matches K-retune 2.891).

| region | mean ms | std | min | note |
|---|---:|---:|---:|---|
| draft_forward_first (step 0) | **18.5** | 3.3 | 8.3 | captured-graph forward ≈ T_compute |
| draft_forward (per K-step) | **67.5** | 4.6 | 63.3 | eager-attn + batch-invariant MoE |
| draft_chain (whole propose) | **94.3** | 8.0 | 81.5 | = 86.1 GPU (18.5 + 67.5) + 8.3 draft-CPU |
| verify (target forward) | **65.4** | 19.9 | 22.8 | full-EP verify incl. real PCIe a2a + DP wait |
| **residual** (cycle − chain − verify) | **23.2** | — | — | sampling/rejection + engine/DP glue |
| **total cycle** | **182.9** | — | — | 94.3 + 65.4 + 23.2 |

- **draft×K = 86.1 ms GPU** (step0 18.5 + one K-step 67.5), + 8.3 ms draft-side CPU.
- **verify = 65.4 ms** (mean; high variance, min 22.8 ≈ the T_verify floor — the
  mean carries the real forced-PCIe all-to-all and the cross-rank DP stall).
- **sampling/rejection + outer CPU = 23.2 ms** (entirely unmodeled).

## 2. Draft cost verdict — did T_compute undercount the draft? YES, by 4.15x

| draft variant (batch 64, 1 tok/seq) | ms | vs T_compute(64) |
|---|---:|---:|
| **T_compute(64) cost-model value** | 16.27 | 1.00x |
| step-0 / captured-graph forward (measured) | 17-18.5 | ~1.1x |
| **full-replica in-chain draft forward (measured)** | **67.54** | **4.15x** |
| skip-A2A *tile* in-chain forward (measured) | 107.36 | 6.60x |
| no-spec / verify-compute forward (measured) | 37.8 | 2.32x |

- **The full-replica draft forward (67.5 ms) is 4.15x T_compute and 1.79x the no-spec
  verify-compute forward (37.8 ms).** The comm-free draft is NOT "genuinely cheaper"
  than the verify compute — it is *more* expensive per forward, because the verify
  runs a single captured full graph while each in-chain draft forward runs eager
  attention + batch-invariant persistent matmuls.
- **T_compute measured the wrong forward.** It matches the step-0 captured-graph
  forward (~17-18 ms) and the phase-24 whole-model SKIP_A2A step (~21 ms at b64),
  i.e. a *single captured* forward — not the K accept-achieving in-chain forwards.
- **The SKIP_A2A tile is NOT a valid cheap-draft stand-in**: at 107 ms it is *1.6x*
  the genuine full-replica draft, because the tile inflates the token count
  (local-chunk repeat to the gathered size) fed to the MoE. The cost model borrowed
  T_compute from the tile's *low-batch intercept*, which happens to land near a cheap
  single captured forward, masking the real in-chain cost.
- Root-cause log line (all 8 ranks, both spec and skiptile):
  `Draft FULL-CG: running the draft chain attention eagerly (non-MLA backend); the
  captured decode graph is not replay-safe for the draft's growing sequence.`

## 3. Term-by-term reconciliation (cost model 64.7 ms -> measured 182.9 ms)

Cost model: `K·T_compute(64) + T_verify(192) = 2·16.27 + 32.20 = 64.75 ms`.
Measured cycle: `182.9 ms`. Gap **118.1 ms**, attributed self-consistently (sums to
the gap exactly):

| term | measured | cost-model | Δ (ms) | share |
|---|---:|---:|---:|---:|
| **(a) draft GPU** (2 fwds: 18.5 + 67.5 = 86.1) | 86.1 | K·T_compute = 32.5 | **+53.5** | **45%** |
| **(b) verify** (region mean) | 65.4 | T_verify(192) = 32.2 | **+33.2** | 28% |
| (c) draft-side CPU (in draft_chain, unmodeled) | 8.3 | 0 | +8.3 | 7% |
| (d) outer CPU / sample-reject (unmodeled) | 23.2 | 0 | +23.2 | 20% |
| **sum** | | | **+118.1** | 100% |

**The draft-compute underestimate dominates (+53.5 ms, 45%)** — the model charged
2·16.3 = 32.5 ms for the draft but the real two forwards cost 86.1 ms. Verify is the
second error (+33.2 ms): the T_verify fit (from a clean microbench) undercounts the
real forced-PCIe verify all-to-all + DP-rank sync at this operating point (tile-verify
with no real collective was only 13.9 ms; the real full-EP verify is 65.4 ms).
**Unmodeled per-cycle CPU orchestration (c+d) is +31.4 ms (27%)** — real but not the
lead term.

## 4. Speedup isolation (baseline no-spec = K-retune 1857.6 tok/s = 34.45 ms/step)

| scenario | cycle ms | speedup |
|---|---:|---:|
| measured | 182.9 | **0.550x** (= reported 0.55x) |
| (a) draft-GPU == K·T_compute (shave 53.5) | 129.4 | 0.777x |
| (b) verify == T_verify (shave 33.2) | 149.7 | 0.672x |
| (c+d) all CPU overhead == 0 (shave 31.4) | 151.5 | 0.664x |
| (all) cost-model cycle 64.7 ms | 64.7 | 1.552x (K-retune baseline) |
| paper formula (baseline = T_verify(64)) | 64.7 | **1.166x (the model's ~1.1x)** |

**Fixing only the draft compute (the dominant term) lifts 0.55x -> 0.78x — still
sub-parity.** No single error, removed alone, crosses 1.0x: the three errors are
comparable in the wall-clock and all must be closed together. The model's ~1.1x
appears only when *all three* are set to zero.

## 5. One-line answer

Comm-free self-spec is ineffective (0.55x) because the **real accept-achieving draft
forward is ~4x more expensive than the cost model's T_compute** — the model fit
T_compute to a single *captured-graph* forward, but on a non-MLA MoE each in-chain
draft step runs **eager attention (the captured graph is not replay-safe for the
draft's growing sequence) plus batch-invariant matmuls**; add the real forced-PCIe
verify all-to-all it also undercounts and ~31 ms of unmodeled per-cycle CPU
orchestration, and the model's 64.7 ms / 1.1x cycle becomes a measured 182.9 ms /
0.55x.

## Files
- Harness `scripts/recon_microbench.py` (spec / nospec / skiptile modes; env-gated
  profiler read), driver `scripts/run_recon.sh`, analysis `scripts/analyze_recon.py`.
- Data `data/rc_qwen30b_b64_K2_{spec,nospec,skiptile}.json`, per-run profile dumps
  `data/rc_profiles_*`, `data/analysis_output.txt`. Logs `logs/`.
- Note: the first skiptile attempt OOM'd on KV-cache allocation at gpu_mem=0.90;
  re-run at 0.85 succeeded. Own workers torn down each run (procs=0 after each).
