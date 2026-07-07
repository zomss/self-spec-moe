# Phase 73 — draft-forward overhead decomposition (FIXED overhead vs PER-TOKEN work)

Source: Phase 72 (`research/72_kernel_trace/results.md`). P72 kernel-traced the
b1 draft forward: ~11 ms steady-state, ~2400 tiny serial kernels (median 1.9
us), ~5 ms real GEMMs + ~6 ms per-layer glue, latency-bound. Open question:
is that ~6 ms glue *reducible vLLM overhead* (amortizes with more tokens/rank,
or fuses away) or a per-token floor? The decisive test is BATCH-SCALING of the
draft forward.

## Objective

Decompose `draft_forward(b_per_rank) = F_fixed + b*m`:
- **F_fixed** (ms): batch-independent engine/latency overhead (amortizes).
- **m** (ms/token): marginal per-token work (irreducible per-token cost).

Then: (2) per-token draft cost vs batch (amortization curve), (3) the GEMM-only
floor from P72 (hard fusion floor at b1), (4) verdict on the user's hypothesis
(fixed-overhead-dominated vs per-token-work) and whether the amortization is
ACCESSIBLE at 16k 2-node MoE-EP serving (pool + wide DP pin per-rank batch low).

## Config

Qwen3-30B-A3B, single-node DP8/EP8 on **h106** (h107/h108 occupied). Self-spec
draft = fp8 full-replica comm-free (`W7_DRAFT_QUANT=fp8 W7_DRAFT_FULL_REPLICA=1
W7_DRAFT_LOCAL_ROUTE=1`) + `VLLM_SELF_SPEC_SHARED_KV=1` +
`VLLM_SELF_SPEC_DRAFT_KV_WINDOW=512` (sinks 16) + P65 flags
(`DP_COORD_CPU CHAIN_LIGHT_MD SKIP_DP_COORD`). **PIECEWISE** draft path
(`CHAIN_PIECEWISE=1`, `DRAFT_FULLCG` OFF — the real / faster chain). K=4.

SHORT context (2k) so per-rank batch scales free of the 16k KV-pool cap; the
windowed draft KV is fixed (cap544, sinks16+window512) and context-independent,
so 2k transfers to 16k. ONE 16k b8 point cross-checks ctx-invariance.

Per-rank decode batch = `W7_BATCHES` (MEASURED: this is a *replicated-batch* DP
harness — each of the 8 ranks independently runs the full batch; confirmed by
steady draft `bs=N`, KV-usage %, and "Running: N reqs"/engine. NOT `W7_BATCHES/8`
as originally assumed.) Sweep W7_BATCHES 1/2/4/8/16/32/64 -> per-rank 1..64.

## Method

`scripts/run_sweep.sh` launches ONE engine per (ctx,batch) point (the
SelfSpecProfiler accumulates per-process and does not reset between batches, so
each batch gets its own `PROFILE_OUT` dir). Coarse profiler (FINE off) records
the always-on regions `draft_forward` / `draft_forward_first` / `verify` /
`draft_chain` with minimal sync perturbation. `scripts/analyze.py` builds the
batch-scaling table + the F_fixed/m least-squares fit + per-token curve + the
16k cross-check. GEMM floor read from P72 `decompose_draft_forward.txt`.

Runs ON h106 (shared node): strict pre-launch foreign-GPU guard (yield if any
non-v-sukmincho compute app appears beyond the permanent ~522 MiB stale ctx);
cleanup only our own `w7_2node.py` / `EngineCore` / `Worker_DP` patterns.

## Decision criteria

- F_fixed >> b1*m (F_fixed is most of the b1 forward) => fixed-overhead-dominated
  (amortizes with batch). m large => per-token-work-dominated (irreducible).
- Accessibility: compare the per-rank batch needed to amortize F_fixed against
  the max per-rank batch reachable at 16k on 2-node MoE-EP (Phase 59/66 pool).

## Artifacts

- `scripts/env_h106.sh`, `scripts/run_sweep.sh`, `scripts/analyze.py`
- `data/prof_ctx<CTX>_b<BATCH>/` per-point profiler dumps; `data/w72n_*.json`
  harness rows; `results.md` (table, fit, floor, verdict).
