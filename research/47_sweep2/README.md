# Phase 47 (sweep2) — definitive speedup-vs-A2A win curve, fully-optimized comm-free self-spec

**Source phase** 42 (comm sweep, pre-piecewise K=4, plateau ~0.82x, no crossover to
1.0x). **What changed since** the genuine full-replica fix (Phase 41) + the PIECEWISE
draft chain (Phase 43, commit `3e98800d3`) + CPU-orchestration cut (Phase 45) took the
draft to ~= `T_compute`; a native K=2 point measured 1.008x at a2a=0. This phase draws
the FULL curve: speedup vs emulated A2A cost, the headline that validates the cost
model in its comm-bound regime.

## Objective

Speedup = spec tok/s / no-spec tok/s across `VLLM_SELF_SPEC_EMULATE_A2A_DELAY_US in
{0,100,250,500,1000}` us/collective x batch {32,64,128}, for the fully-optimized
comm-free self-spec (piecewise ON, K=2). Six deliverables: (1) speedup table +
crossover, (2) A2A->f mapping + speedup-vs-f vs cost-model ideal
`accept_len/(K(1-f)+1)`, (3) delta vs pre-piecewise (Phase 42 K=4), (4) draft-shielding
check (real=0 collectives? still charged the sleep?), (5) multi-node projection at
f~0.6-0.8, (6) sanity 1.008x @ a2a0 b64.

## Config

Model `Qwen/Qwen3-30B-A3B`, DP8/EP8, tp=1, forced-PCIe
(`NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=1`), FP8 full-replica
comm-free draft. Fully-optimized stack:
`VLLM_SELF_SPEC_DRAFT_FULL_REPLICA=1 DRAFT_LOCAL_ROUTE=1 DRAFT_FULL_CG=1
COMPILE_CONSISTENT=1 DRAFT_CHAIN_PIECEWISE=1`, K=2, greedy.
`VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0`. Harness
`research/34_worldA_system/scripts/w7_fp8_timing.py` UNCHANGED (two-length slope,
OUTLEN=160/SHORTLEN=32, CUDA graphs ON, WARMUP=2, ITERS=3, greedy seed=0 ignore_eos).
Spec engines strictly serial; OWN workers torn down after each run. HEAD `579d728bf`.

## Baselines (REUSED)

No-spec tok/s from Phase 42 (`research/42_comm_sweep/data`, unchanged by piecewise).
Sanity-checked: no-spec b64 a2a0 = 1857.6 (in the 1858-1866 band). Pre-piecewise K=4
spec also from Phase 42 for the delta.

## Commands

```
bash research/47_sweep2/scripts/run_sweep2.sh                 # spec sweep (piecewise K=2)
A2A_US=500 bash -c '... probe_shield.py'                      # draft-shielding probe
.venv/bin/python research/47_sweep2/scripts/analyze_sweep2.py # tables + curves
```

## Decision criteria

Crossover where speedup passes 1.0x (expect ~0 / everywhere given the 1.008x native
point). Report measured AND draft-shielded curve; compare measured/ideal ratio.

## Artifacts

`data/w7fp8_sweep2_a2a*_fp8_spec_cg*_K2.json` (spec), `data/probe_counts/` (shielding),
`results_W7_sweep2.md` (the win curve).
