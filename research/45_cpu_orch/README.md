# Phase 45: CPU-orchestration reduction for the comm-free self-spec cycle

**Source phase:** 44 (recon; decomposed the 182.9 ms Qwen3-30B K=2 b64 cycle into
draft-GPU 86 ms, verify-GPU 65 ms, and ~31.4 ms CPU orchestration = draft-side
8.3 ms + sampling/rejection/outer residual 23.2 ms). HEAD `753f9f91a`.

**Objective.** Remove per-cycle CPU-orchestration overhead from the comm-free
self-spec (draft_model) decode path, env-gated, with ZERO change to
acceptance/output (lossless). Target the 31.4 ms: host<->device syncs
(`.item()`/`.tolist()`/`.cpu()`), per-step tensor allocs, redundant metadata
rebuilds, per-sequence Python loops.

**Config.** `Qwen/Qwen1.5-MoE-A2.7B` (fast iterate) then `Qwen/Qwen3-30B-A3B`
(confirm), DP=8/EP=8, forced-PCIe (`NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0
NCCL_IB_DISABLE=1`), FP8 full-replica comm-free draft (`DRAFT_LOCAL_ROUTE=1
DRAFT_FULL_REPLICA=1 DRAFT_FULL_CG=1 COMPILE_CONSISTENT=1`), K=2, batch 64,
greedy, CUDA graphs ON. `VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0`.
`.venv/bin/python`, `PYTHONPATH=/data/smcho/ssm-cpu`.

**Method.**
1. **Fine CPU profile** — extend the self-spec profiler with a non-CUDA-syncing
   `cpu_region()` and instrument the outer CPU sites (rejection sample, output
   parse, bookkeeping loop, next-input build, `_prepare_inputs`) plus the
   existing per-step draft regions. `scripts/cpu_profile.py` (`CP_JOB=profile`).
2. **Reduce** — env-gated behind `VLLM_SELF_SPEC_CPU_ORCH` (master) /
   `VLLM_SELF_SPEC_FAST_PARSE`; default off => byte-identical.
3. **Validate** — `CP_JOB=accept` dumps deterministic per-request output token
   ids; before/after diff proves losslessness. `scripts/cycle_speedup.py`
   (prefill-cancelled two-length slope) reports cycle-ms + speedup vs no-spec.

**Scripts.** `cpu_profile.py` (fine profile + accept dump), `run_ab.sh` (base vs
orch A/B + losslessness diff), `cycle_speedup.py` (nospec/spec_base/spec_orch
cycle + speedup), `pyspy_run.py` (long steady decode for external profiling).
Data in `data/`, logs in `logs/`. Results in `results_W7_cpu_orch.md`.

**Key finding (see results).** The sampling/rejection/bookkeeping CPU the recon
lumped into the 23.2 ms residual is <1 ms/cycle — already well-optimized (the
padded-batch path keeps tokens on-device; prior sample-in-graph work). The bulk
of the residual is engine-loop (scheduler `schedule`/`update_from_output`
per-request loops in the EngineCore process) + the per-step DP cross-rank
sync barrier (forced-PCIe), which is comm-wait, not self-spec-scoped reducible
CPU. The self-spec-scoped lossless cuts (fast non-blocking rejection parse,
persistent chain buffers) are applied and measured.
