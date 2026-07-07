# Phase 72 — the confirming KERNEL-LEVEL trace of the full-CG draft forward

Source: Phase 71 (`research/71_moe_cudagraph_investigation/FINDINGS.md`). P71
concluded from code + arithmetic that the ~25 ms draft forward is NOT weight
traffic and NOT a dense MoE, but a **latency floor of a deep b1 serial
small-kernel chain**, with MoE small and sparse. The one piece P71 flagged as
unmeasured: an actual kernel-level trace of the FULL-CG draft **replay** to
prove the latency-floor signature and give the MoE-vs-attention split.

## Objective

Capture a kernel-level trace of steady-state decode on rank 0 for the Phase-70
full-CG draft arm, isolate ONE draft-chain forward (one of the K=4 b1 steps),
and answer:

1. **Latency-floor signature:** GPU-active vs wall vs idle/gap fraction;
   #kernels; median kernel + inter-kernel gap.
2. **Category split:** MoE vs attention vs dense linear vs norm/rope vs fp8
   quant vs KV write. Headline = MoE-vs-attention share.
3. **Sanity vs arithmetic:** is the MoE expert GEMM consistent with SPARSE
   top-8 (~1-2 GB / ~1 ms), i.e. a small share — or a fat dense MoE (refuting
   P71)?
4. Is the P69 scratchpad SDPA (fixed-shape cap544) itself a notable cost?

## Config traced

Phase-70 full-CG draft arm, single node h107 (DP8/EP8), 16k
(`W7_CTX_TOKENS=16384 W7_MAX_MODEL_LEN=20480`): fp8 full-replica comm-free draft
(`W7_DRAFT_QUANT=fp8 W7_DRAFT_FULL_REPLICA=1 W7_DRAFT_LOCAL_ROUTE=1`) +
`VLLM_SELF_SPEC_SHARED_KV=1` + `VLLM_SELF_SPEC_DRAFT_KV_WINDOW=512` (sinks 16) +
`VLLM_SELF_SPEC_DRAFT_FULLCG=1` + P65 flags (`DP_COORD_CPU`, `CHAIN_LIGHT_MD`,
`SKIP_DP_COORD`), W512, K4, batch 8 (-> b1/rank), chat + on-dist prompts.

## Method

Tool: **torch profiler (kineto) Chrome trace**, rank-0 only, via the existing
P64/P65 harness `w7_trace16k.py` + `VLLM_CUSTOM_SCOPES_FOR_PROFILING=1`.
`nsys` was not used: it would attach to all 8 forked DP workers (isolating just
rank 0's few cycles needs `cudaProfilerApi` capture-range surgery in the
harness), whereas the kineto path profiles ONLY rank 0, already exists, and —
verified here — records the **cudagraph-internal kernels** of a FULL-CG replay
(319k GPU kernels on the main stream), which is exactly what the decomposition
needs. This is the mission's sanctioned fallback.

REPLAY was confirmed: all 8 ranks logged
`[fullcg-dbg] ... REPLAY ctx_desc=BatchDescriptor(num_tokens=1, num_reqs=1,
uniform=True)` on the b1 draft FULL wrapper during steady decode — genuine full
replay, not eager fallback.

### Isolating ONE draft forward

There is no per-chain-step annotation (`gpu_model_runner: draft` wraps the whole
K=4 chain, and its GPU projection bleeds across async boundaries). So the whole
GPU timeline is segmented into 48-layer forwards using the per-layer
`reshape_and_cache_flash` landmark (exactly 1/layer, used by both draft and
target), cutting forward boundaries at the largest inter-kernel gap between
layer-47 and the next layer-0 (drops the inter-step sample/window-build).
Each forward is classified **draft** (math-backend scratchpad SDPA) vs
**target-verify** (has the `FlashAttnFwd` kernel). Steady-state pattern is
`TDDDD` = 1 target verify + K=4 draft forwards per cycle.

## Scripts

- `scripts/run_trace.sh` — single-node DP8 arm-B + FULLCG + rank-0 kineto trace.
- `scripts/inspect_trace.py` — dump trace structure (streams, annotations).
- `scripts/decompose.py` — segment, classify, decompose ONE steady draft forward.

Results: `results.md`. Data (text summaries committed; raw trace gitignored):
`data/decompose_draft_forward.txt`, `data/per_forward_timeline.txt`,
`data/trace_fullcg_w512k4_b8/*.pt.trace.json.gz` (15 MB, gitignored).
