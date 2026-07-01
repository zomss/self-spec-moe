# Phase 46: top-C draft-prune elasticity probe (W7 / W2b gate)

**Source phase:** 45 (CPU-orch cut) / 44 (recon). HEAD `3e98800d3` (genuine
comm-free full-replica draft + piecewise chain). Worktree `/data/smcho/ssm-topc`,
branch `w7-topc-probe`.

**Objective.** Measure the acceptance-vs-#experts elasticity of the comm-free
self-spec draft: if the DRAFT computes only the **top-C** highest-gate-weight
experts per token (C < top_k, renormalized over C) instead of the full top_k,
how fast does accept_len fall, and how much draft compute is saved? This is a
cheap probe that gates whether the **W2b bounded expert-cache** build is worth
it. Lossless regardless of C (the full-top_k VERIFY corrects); only accept_len
(-> speedup) is at stake.

**Implementation (env-gated, draft-only, default off).**
`VLLM_SELF_SPEC_DRAFT_TOPC` (int, 0=off). When >0 AND the draft flag is active
(`self_spec_local_route`), the DRAFT MoE keeps the C largest-gate-weight experts
after `select_experts`, renormalizes those C to sum to 1, and narrows the
`topk_ids`/`topk_weights` tensors to width C so the fused kernel computes only C
experts/token. VERIFY (no draft flag) keeps full top_k -> lossless.
Files: `vllm/model_executor/layers/fused_moe/local_route.py` (`prune_topk_to_topc`),
`vllm/model_executor/layers/fused_moe/runner/moe_runner.py` (call + confirm log),
`vllm/envs.py` (the env var). Modular-kernel path only; monolithic warns once.

**Config.** Two models: `Qwen/Qwen3-30B-A3B` DP8 (top_k=8) and
`Qwen/Qwen1.5-MoE-A2.7B` DP8 (top_k=4). Piecewise-on trio
(`DRAFT_FULL_REPLICA=1 DRAFT_LOCAL_ROUTE=1 DRAFT_FULL_CG=1 COMPILE_CONSISTENT=1
DRAFT_CHAIN_PIECEWISE=1`), forced-PCIe (`NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0
NCCL_IB_DISABLE=1`), K=2, greedy, batch 64, FP8 draft. `VLLM_USE_DEEP_GEMM=0
VLLM_MOE_USE_DEEP_GEMM=0`. `/data/smcho/self-spec-moe/.venv/bin/python`,
`PYTHONPATH=/data/smcho/ssm-topc`.

**Measurements.**
1. accept_len(C), C = top_k .. 1 (C=top_k reproduces the full-replica accept).
2. draft_forward mean_ms(C) (env profiler) -> g(C)=draft_ms(C)/draft_ms(top_k),
   its floor (attention/router/dense don't shrink).
3. losslessness: at one C, output tokens byte-identical to no-spec.

**Projection.** speedup(C,f) = accept_len(C) / (K*g(C)*(1-f)+1), K=2, at
f in {0.4, 0.62, 0.8}. Optimal C per f; does any C beat C=top_k?

**Scripts.** `scripts/topc_sweep.py` (one (model,C) engine), `scripts/run_topc.sh`
(sweep driver), `scripts/run_lossless.sh` + `scripts/diff_tokens.py` + `diff3.py`
(losslessness), `scripts/analyze_topc.py` (tables + projection). Data in `data/`,
logs in `logs/`, results in `results_W7_topc.md`.

**Verdict.** No — the accept/compute tradeoff does not justify W2b. g floors high
(0.66 Qwen3-30B, 0.86 Qwen1.5-MoE: attention/router/dense dominate the draft, only
the expert GEMM shrinks). Qwen1.5-MoE: pruning strictly worse at every f. Qwen3-30B:
best case +1.9% (C=6 @ f=0.4), 0% at f=0.8, within g-noise. Losslessness confirmed:
spec C=4 == spec C=8 byte-for-byte (the spec-vs-no-spec mismatch is the pre-existing
bf16 comm-free divergence, orthogonal to top-C). See `results_W7_topc.md`.
