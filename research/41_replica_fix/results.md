# Phase 41: Bug B fixed — the comm-free full-replica draft is now GENUINE

**Branch** `w7-num-divergence`, worktree `/data/smcho/ssm-num`. The last bug blocking
a real comm-free MoE draft replica is fixed and the structural large-EP accept
collapse is eliminated. Forced-PCIe, `FULL_CG=1 COMPILE_CONSISTENT=1`,
`VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0`, greedy.

## 1. The fix (file:line)

`vllm/model_executor/layers/fused_moe/config.py`, `FusedMoEParallelConfig.make`
(after the dp/pcp rank setup, before `flatten_tp_across_dp_and_pcp`). New
full-replica branch, gated on `VLLM_SELF_SPEC_DRAFT_FULL_REPLICA`:

```python
if (not use_ep and dp_size_ * pcp_size_ > 1
        and envs.VLLM_SELF_SPEC_DRAFT_FULL_REPLICA
        and not vllm_parallel_config.enable_expert_parallel):
    return FusedMoEParallelConfig(
        tp_size=tp_size_,                 # =1 in pure-DP self-spec -> NO flatten-TP
        tp_rank=0 if tp_size_ == 1 else get_tensor_model_parallel_rank(),
        ep_size=1, ep_rank=0, dp_size=dp_size, ...
        use_ep=False, ...)
```

Before the fix, the non-EP branch ran `flatten_tp_across_dp_and_pcp` ->
`tp_size = dp_size*pcp_size*tp_size = 8` at DP=8, TP-sharding each expert
(intermediate 2816->352) and relying on `tensor_model_parallel_all_reduce`
(`runner/moe_runner.py:_maybe_reduce_final_output`) to recombine. In pure-DP
`get_tp_group().world_size == 1` -> that all-reduce is a no-op -> 1/8-sharded
output. The fix forces `tp_size=1, ep_size=1`: full unsharded experts resident AND
fully computed locally; `_maybe_reduce_final_output`'s `(tp_size>1 or ep_size>1)`
guard is now False -> NO all-reduce attempted -> comm-free per-rank replica.
(Also added `import vllm.envs as envs` to `config.py`.) Default off -> unchanged.

## 2. Gate 1 — replica is genuine (rel-err ~0.0 vs plain MoE at DP=8)

`scripts/dump_moe.py` (config A2 = full-replica, sets DRAFT_FULL_REPLICA=1; D =
plain MoE) + `scripts/analyze_gate1.py`. Reference = single-GPU plain MoE (D, DP=1),
norm 7.2933.

| compare (A=replica, D=plain) | norm | ref_norm | rel_mean | verdict |
|---|---:|---:|---:|---|
| A_dp1 vs D_dp1 (sanity)      | 7.2933 | 7.2933 | 0.0 | == |
| **A_dp8 vs D_dp1 (GATE 1)**  | **7.2933** | **7.2933** | **0.0** | **bit-identical** |
| D_dp8 vs D_dp1 (plain DP noise) | 7.2939 | 7.2933 | 4.0e-2 | plain-MoE's own DP noise |

Post-fix the full-replica draft MoE output at DP=8 is **bit-identical** to plain MoE
(rel-err 0.0, ratio 1.0000). Pre-fix it was norm 5.12 (0.70x, rel 0.539).

Structural (`scripts/probe_draft_weight_shape.py`, `probe_draft_tp_group.py`),
DP=8, every rank:
`use_ep=False, ep_size=1, tp_size=1, expert_map=None, local_num_experts=60/60,
w13_weight=(60, 2816, 2048)` — FULL unsharded experts (was tp_size=8, sharded
intermediate 352, all-reduce over 1 rank).

## 3. Gate 2 — accept recovers; the DP collapse is gone (flat ~5.0)

`research/40_num_divergence/scripts/accept_run.py` CFG=A (full-replica), K=4, greedy.

| DP | accept_len BEFORE | accept_len AFTER | per-position rate (after) |
|---:|---:|---:|---|
| 2 | 4.88 | **4.96** | [1.00, 0.99, 0.99, 0.98] |
| 4 | (~3.x) | **4.92** | [1.00, 0.98, 0.97, 0.97] |
| 8 | **2.18** | **4.92** | [0.995, 0.976, 0.974, 0.971] |
| 8 (C, EP-full upper bound) | 5.00 | 5.00 | [1, 1, 1, 1] |

The 4.88->2.18 DP-width collapse is eliminated: accept is now **flat ~4.92-4.96**
across DP=2/4/8, within 0.08 of the EP-full upper bound (5.00). The small residual
(0.08) is the genuine bf16 reduce-structure difference (comm-free local-sum vs EP
cross-rank reduce), NOT the structural sharding bug.

## 4. Gate 3 — still comm-free + lossless

- **Comm-free:** the draft MoE is `use_ep=False, ep_size=1, tp_size=1` on every rank
  (Gate 1 probes) -> the non-EP MoE path never constructs
  `MoEPrepareAndFinalizeNaiveDPEPModular` / `get_ep_group().dispatch` and
  `_maybe_reduce_final_output` issues no all-reduce. Comm-free by construction (no
  all-to-all on the draft step at all). Only the full-EP verify pays the EP reduce.
- **Lossless:** `scripts/verify_lossless_cfg.py`, DP=8. Full-replica draft (A)
  output is **16/16 prompts, 1024/1024 tokens IDENTICAL** to the EP-full-draft
  reference (C) — the canonical correct spec-decode output. (Spec-vs-plain-nospec
  has the documented batched-verify near-tie caveat; config C exhibits it
  identically, so it is independent of this fix.)

## 5. Headline — Qwen3-30B-A3B, DP=8/EP=8, FP8 full-replica draft

`research/34_worldA_system/scripts/w7_fp8_timing.py`, two-length-slope decode tok/s,
forced-PCIe, FULL_CG+COMPILE_CONSISTENT.

**Accept (DP=8, FP8 draft, K=3):** accept_len **~3.78-3.82** (batch 16/64/128). The
structural collapse is GONE; the residual gap to 5.0 is the genuine FP8 quant-accept
gap (the draft's FP8 experts diverge slightly from the bf16 verify), as anticipated.

**Native forced-PCIe decode tok/s (a2a_us=0):**

| batch | nospec tok/s | spec (FP8) tok/s | accept_len | speedup |
|---:|---:|---:|---:|---:|
| 16  | 748.6  | 302.1 | 3.82 | 0.40x (suspect) |
| 64  | 1876.3 | 846.3 | 3.79 | 0.45x |
| 128 | 2688.7 | 1154.6 | 3.78 | 0.43x |

**Comm-bound regime (emulated exposed A2A = 100 us/collective, batch=64, K=3):**

| | nospec tok/s | spec (FP8) tok/s | accept_len | speedup |
|---|---:|---:|---:|---:|
| native (a2a_us=0)   | 1876.3 | 846.3 | 3.79 | 0.45x |
| a2a 100us/collective | 1467.9 | 912.2 | 3.79 | **0.62x** |

**Wall-clock speedup is < 1.0x on this single-node box** even with the fixed draft.
The reason is compute, not the fix: the full 30B-replica draft run K=3 times per
step costs more than the EP all-to-all it saves. The exposed-A2A penalty hurts
no-spec (1876->1468) and lifts the relative speedup (0.45x->0.62x), but on a single
node the all-to-all does not dominate enough to cross 1.0x. This matches phase 34's
structural conclusion: a wall-clock win needs the all-to-all to dominate (genuinely
comm-bound multi-node fabric) AND a cheap draft. The fix removes the accept collapse
(2.18->4.92 bf16; FP8 ~3.78) but not the single-node compute imbalance — the
comm-free saving is real but smaller than the full-replica draft's compute on this
hardware.

## Verdict
The structural Bug B is fixed: the full-replica draft is now a GENUINE per-rank
comm-free replica (rel-err 0.0 vs plain MoE, full unsharded experts), accept is flat
~4.92 across DP=2/4/8 (was 2.18 at DP=8), comm-free + lossless vs the EP-full
reference. The headline wall-clock speedup is hardware-bound (single-node, comm not
the bottleneck), not a correctness/structure issue.
