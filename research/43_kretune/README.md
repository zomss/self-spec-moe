# Phase 43: K-retune — best-K speedup curve for the comm-free full-replica draft

**Source phase:** 42 (A2A comm sweep, K=4). HEAD `ee778e3f5`.

**Objective.** Phase 42 found K=4 never crosses 1.0x within 1000us A2A (measured
plateau ~0.82x, b64) and that K=4 was WORSE than K=3 (K=4 gave 0.36/0.57 at a2a
0/100us b64; a prior K=3 run gave ~0.45/0.62). The limiter looked like draft compute,
not comm. This retune measures **K in {2,3}** to find the best-K speedup curve and
whether a smaller K (fewer draft forwards -> cheaper cycle, but lower accept_len)
crosses 1.0x at a realistic A2A cost.

**Config.** `Qwen/Qwen3-30B-A3B`, DP=8/EP=8, forced-PCIe
(`NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=1`), FP8 full-replica
comm-free draft (`DRAFT_FULL_REPLICA=1 DRAFT_LOCAL_ROUTE=1 DRAFT_FULL_CG=1
COMPILE_CONSISTENT=1`), greedy, `VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0`.
Two-length-slope decode (OUTLEN=160, SHORTLEN=32), CUDA graphs ON, WARMUP=2, ITERS=3.
Harness reused UNCHANGED: `research/34_worldA_system/scripts/w7_fp8_timing.py`.
NO vllm/ changes.

**Sweep.** K in {2,3} x `VLLM_SELF_SPEC_EMULATE_A2A_DELAY_US in {0,100,250,500}`
us/collective, batch 64. **No-spec baselines reused** from Phase-42
(`research/42_comm_sweep/data`, K-independent — no re-run). K=4 spec reused likewise.
speedup = spec/no-spec at each (K, delay). Serial engines; own workers torn down
between runs.

**Draft-shielding caveat (carried over from Phase 42).** The full-replica draft's MoE
routes through `naive_dp_ep` dispatch/combine, so the injected A2A sleep is NOT gated
by local-route -> the draft ALSO eats the sleep, OVER-charging spec (measured speedup
is a pessimistic lower bound vs a true multi-node where the draft pays 0 A2A). We
report BOTH the measured and a draft-shielded projection (draft pays 0 A2A; verify
slope = s_ns/accept_len) + the shielded crossover delay per K.

**Decision criterion.** Best K at each delay; does any K cross 1.0x at a realistic
A2A cost (30-200us)? Shielded crossover per K. Verdict: does K-retune rescue the win
at realistic comm, or confirm the draft-compute bound (best K still <1.0x at <=200us)?

**Scripts.** `scripts/run_kretune.sh` (driver, spec-only), `scripts/analyze_kretune.py`
(measured + shielded tables, best-K, crossover). Data in `data/` (K2/K3 spec JSONs);
no-spec + K4 spec reused from `../42_comm_sweep/data/`. Logs in `logs/`. Results in
`results_W7_kretune.md`.
