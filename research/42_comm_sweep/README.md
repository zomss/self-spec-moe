# Phase 42: A2A-cost comm sweep — where does the comm-free full-replica draft win?

**Source phase:** 41 (Bug B fixed: genuine comm-free full-replica draft; accept
recovered to ~3.78 FP8 at DP=8, lossless). HEAD `dd842e17e`.

**Objective.** World A's comm-free full-replica draft is correct and lossless but
wall-clock speedup on a single node is < 1.0x (0.45x native, 0.62x at 100us-A2A,
batch 64) because the 30B-replica draft's K-forward compute exceeds the all-to-all
it saves WHEN COMM IS CHEAP. Emulate increasing A2A cost (stand-in for a multi-node
fabric where the all-to-all dominates) to find the crossover where spec passes 1.0x.

**Config.** `Qwen/Qwen3-30B-A3B`, DP=8/EP=8, forced-PCIe
(`NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=1`), FP8 full-replica
comm-free draft (`DRAFT_FULL_REPLICA=1 DRAFT_LOCAL_ROUTE=1 DRAFT_FULL_CG=1
COMPILE_CONSISTENT=1`), K=4, greedy, `VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0`.
Two-length-slope decode throughput (OUTLEN=160, SHORTLEN=32), CUDA graphs ON,
WARMUP=2, ITERS=3. Harness reused UNCHANGED:
`research/34_worldA_system/scripts/w7_fp8_timing.py`.

**Sweep.** `VLLM_SELF_SPEC_EMULATE_A2A_DELAY_US in {0,100,250,500,1000}` us/collective
x batch {32,64,128}, measuring spec tok/s and no-spec tok/s at each; speedup =
spec/no-spec. Serial engines (one DP group at a time; own workers torn down between).

**Decision criterion.** Crossover delay where speedup == 1.0x per batch (interpolate).
Is that a realistic multi-node all-to-all cost? Projected speedup at 300-500us.

**Scripts.** `scripts/run_sweep.sh` (driver), `scripts/analyze_sweep.py` (tables +
crossover), `scripts/probe_a2a_counts.py` (fairness probe). Data in `data/`, logs in
`logs/`. Results in `results_W7_comm_sweep.md`.
