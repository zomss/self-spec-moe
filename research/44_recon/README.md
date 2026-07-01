# Phase 44: recon — why comm-free self-spec's 0.55x wall-clock != the model's ~1.1x

**Source phase:** 43 (K-retune; 0.55x @ K=2 b64 a2a=0). HEAD `11f4777fe` (genuine
comm-free full-replica draft, tp=ep=1).

**Objective.** Explain, term by term, why the comm-free self-spec real wall-clock
(0.55x) differs from the C1 cost-model prediction (~1.1x). The cost model is
`speedup = E_tok(β,k)·T_verify / (k·T_compute + T_verify)` with
`T_compute(T)=15.3+0.0152T` (comm-free draft, SKIP_A2A tile fit) and
`T_verify(T)=22.7+0.0495T` (full-EP verify), from `../paper_section_C1_analysis.md`
and `../24_commbound_throughput/`. It omits per-cycle CPU orchestration. Measurement
only — no `vllm/` changes.

**Config.** `Qwen/Qwen3-30B-A3B`, DP=8/EP=8, forced-PCIe (`NCCL_P2P_DISABLE=1
NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=1`), FP8 full-replica comm-free draft
(`DRAFT_FULL_REPLICA=1 DRAFT_LOCAL_ROUTE=1 DRAFT_FULL_CG=1 COMPILE_CONSISTENT=1`),
K=2, batch 64, greedy, a2a delay 0, CUDA graphs on. `VLLM_USE_DEEP_GEMM=0
VLLM_MOE_USE_DEEP_GEMM=0`. `.venv/bin/python`.

**Three measurements.**
1. **Real spec cycle** (profiler `VLLM_SELF_SPEC_PROFILE=1`): decompose into
   draft×K, verify, sampling/rejection, residual CPU orchestration; cross-check the
   total against the K-retune 1021 tok/s (cycle ~181 ms).
2. **Draft cost** (critical): the actual full-replica comm-free draft forward at
   batch 64 vs `T_compute(64)=16.27 ms`, vs a `SKIP_A2A` tile forward, vs the
   no-spec/verify forward (~34.5 ms). Does the accept-achieving draft cost *more*
   than T_compute? By how much?
3. **Verify** forward over B·(K+1)=192 tokens vs `T_verify(192)=32.2 ms`.

**Reconcile.** Build the term table: cost model `2·16.27 + 32.2 = 64.7 ms` vs
measured cycle ~181 ms; attribute the gap to (a) draft underestimate, (b) verify
underestimate, (c/d) unmodeled CPU. State the dominant term; isolate each error's
effect on the speedup.

**Modes** (`scripts/recon_microbench.py`, one DP=8 engine each, serial with own
teardown):
- `spec`    — FP8 full-replica comm-free draft (the real cycle + draft forward).
- `nospec`  — plain full-EP decode (verify/no-spec compute reference).
- `skiptile`— EP-shard draft + `SKIP_A2A=1`, `DRAFT_LOCAL_ROUTE=0` (the T_compute
  tile draft forward).

**Scripts.** `scripts/run_recon.sh` (driver), `scripts/recon_microbench.py`
(harness), `scripts/analyze_recon.py` (reconciliation table + speedup isolation).
Data in `data/`, logs in `logs/`, results in `results_W7_recon.md`.

**Verdict.** The draft-compute underestimate dominates the gap (45%): the real
in-chain draft forward is 4.15x T_compute because it runs eager attention (non-MLA,
not replay-safe) + batch-invariant matmuls, none of which T_compute (a single
captured forward) measured. See `results_W7_recon.md`.
