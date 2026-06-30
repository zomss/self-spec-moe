# W7 num-divergence: the REAL cause of the comm-free full-replica accept collapse

**Question (the sharp contradiction).** A genuine full-replica comm-free draft
(`DRAFT_FULL_REPLICA=1 DRAFT_LOCAL_ROUTE=1`) should hold ALL experts on every
rank, route to the full top-k locally, and compute the SAME MoE as the EP verify
→ accept ~5.0. The verify IS faithful (≈ plain MoE, rel 0.003) even at DP=8. Yet
measured accept (Qwen1.5-MoE, DP=8, bf16, K=4, greedy) is **2.18** at DP=8 vs
**4.88** at DP=2. A genuine full replica cannot accept 2.18. Which premise is
false?

**Setup.** `Qwen/Qwen1.5-MoE-A2.7B` (60 experts top-4, 24 layers, non-MLA), DP→EP,
forced-PCIe, greedy, K=4, bf16, `VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0`,
`DRAFT_FULL_CG=1 COMPILE_CONSISTENT=1`. Worktree `/data/smcho/ssm-num`, branch
`w7-num-divergence`.

---

## TL;DR — the false premise

**The premise "the draft is a genuine full replica" is FALSE.** Phase 40's earlier
"config A" probe used the MAIN model with `enable_expert_parallel=False` — a true
full replica. But the ACTUAL self-spec DRAFT under `DRAFT_FULL_REPLICA=1` is **not**
full-replica at any DP>1. There are TWO independent bugs in the draft build, both
in how the draft's MoE parallel state is derived:

1. **Bug A — the override is discarded (coverage loss).** The draft was built
   `use_ep=True`, EP-sharded to **7–8 of 60 experts/rank** at DP=8 (30/60 at DP=2),
   NOT a 60/60 replica. With `LOCAL_ROUTE=1`, the router masks each token's top-4
   to the rank's tiny shard → most top-4 experts are masked away → wrong, smaller
   MoE output. This scales exactly with DP: 30/60 (DP=2, accept 4.88) →
   7–8/60 (DP=8, accept 2.18). **This is "coverage loss masquerading as full
   replica."**

2. **Bug B — flatten-TP without a TP group (sharded, un-reduced output).** After
   fixing Bug A so the draft is genuinely `use_ep=False` with all 60 experts
   resident, the non-EP MoE path sets `tp_size = flatten(dp_size·tp_size) = 8` and
   TP-shards each expert (intermediate 2816→**352**), expecting a final
   `tensor_model_parallel_all_reduce` over 8 ranks. But in a pure-DP deployment
   `get_tp_group().world_size = 1`, so the all-reduce is a **no-op** → each rank's
   draft MoE output is only **1/8 of the expert GEMM** → still wrong. Accept stays
   **2.180952…** (bit-identical to pre-fix), proving fixing coverage alone does
   nothing while Bug B persists.

The verify is untouched and stays faithful (EP-sharded, full experts per shard,
real EP reduce). The drift is entirely on the DRAFT side.

---

## Step 1 — is the draft GENUINELY full-replica at DP=8? NO.

Per-rank DRAFT MoE state (`probe_draft_vs_verify_state.py`; the draft forward is
tagged by the local-route key, the verify is not):

| | use_ep | expert_map | local_num_experts | w13 intermediate |
|---|---|---|---|---|
| **DRAFT** (pre-fix), DP=8, rank0 | **True** | not None | **8** | 2816 |
| DRAFT (pre-fix), DP=8, rank7 | True | not None | **7** | 2816 |
| DRAFT (pre-fix), DP=2, both ranks | True | not None | **30** | 2816 |
| VERIFY, DP=8 | True | not None | 7–8 | 2816 |

The draft is an EP shard identical to the verify — NOT a replica. Resident-expert
count tracks DP exactly: DP=2→30/60, DP=4→15/60, DP=8→7–8/60.

After Bug-A fix (set draft config current at load): DRAFT now
`use_ep=False, expert_map=None, local_num_experts=60, nonzero=60/60` on every rank
→ genuine full coverage. BUT `w13` intermediate is **352** (= 2816/8), i.e.
TP-sharded (Bug B).

## Step 2 — DP scaling: the drift is on the DRAFT, the verify stays faithful

DP-sweep of config-A MoE output norm (full-replica main-model proxy, same 47-token
prompt, `data/dp_sweep/`), reference = plain MoE (D) norm 7.29:

| | DP=1 | DP=2 | DP=4 | DP=8 |
|---|---:|---:|---:|---:|
| A (full-replica) norm | 7.29 | 6.09 | 5.40 | 5.12 |
| ratio to plain-MoE | **1.000** | 0.835 | 0.741 | 0.703 |
| D (plain) norm | 7.29 | 7.29 | — | 7.29 |

At **DP=1 the full-replica output is bit-identical to plain MoE** (rel-err 0.0).
The shrinkage appears only at DP>1 and worsens monotonically — a DP-dependent drift
on the draft side. The verify ↔ plain-MoE stays rel 0.003 at every DP (phase 40
Step 1). So the divergence is entirely the DRAFT's DP-derived MoE config.

Accept (end-to-end ground truth):

| config | accept_len | per-position rate [p0,p1,p2,p3] |
|---|---:|---|
| A DP=2 | **4.88** | (30/60 resident → most top-4 land local) |
| A DP=8 pre-fix | **2.18** | [0.41, 0.29, 0.25, 0.22] |
| A DP=8 post Bug-A fix | **2.18** | [0.41, 0.29, 0.25, 0.22] (Bug B blocks) |
| C DP=8 (EP verify) | 5.00 | [1,1,1,1] |

Rejection starts at **position 0** (pos-0 rate 0.41) → a per-step MoE divergence at
depth 0, NOT KV drift. Consistent with a wrong draft MoE output, not coverage decay
with depth.

## Step 3 — the decisive dump: draft MoE vs verify MoE differ, structurally

Per-token: routing (router logits, top-4 ids, weights) is **bit-identical** A vs
verify (phase 40 Step 1), but the post-MoE OUTPUT diverges (rel 0.539, norm
5.12 vs 7.29) with the divergence growing with DP. At the deepest reduce
(`TritonExperts.moe_sum`, `probe_moesum_slots.py`) all 4 top-k slots are non-zero
(`frac_zero=[0,0,0,0]`) but each slot's magnitude is uniformly smaller at DP=8 —
i.e. the per-expert GEMM contributions themselves are smaller (Bug A: masked to the
wrong/weaker resident experts; Bug B: each GEMM is a 1/8 TP shard). The dispatch /
combine are pure pass-through under local-route (`GATHERED=False, REDUCED=False`) —
the comm-free skip works; the bug is upstream in the draft's MoE parallel state.

## Step 4 — the responsible ops (file:line)

- **Bug A — override discarded.**
  `vllm/config/speculative.py:1002` correctly sets the draft
  `ParallelConfig.enable_expert_parallel=False` for FULL_REPLICA, BUT
  `vllm/v1/spec_decode/llm_base_proposer.py:_create_draft_vllm_config` built the
  draft from the TARGET `vllm_config` (EP=True) and never applied the override; and
  the FusedMoE reads the GLOBAL `get_current_vllm_config().parallel_config`
  (`vllm/model_executor/layers/fused_moe/layer.py:209,221`), not the config passed
  to `get_model`. So the draft inherited the target's EP=True →
  `FusedMoEParallelConfig.make` (`config.py:1188`) set `use_ep=True`. Proof:
  `probe_draft_config_discarded.py` → `draft_parallel_config.enable_expert_parallel
  = False` but `_create_draft_vllm_config USED ... = True`.
  **Fix applied:** propagate `enable_expert_parallel=False` onto the draft config
  AND set it current via `set_current_vllm_config(draft_vllm_config)` around
  `get_model` in `_create_draft_vllm_config` / `_get_model`. Verified: draft now
  60/60 experts on every rank.

- **Bug B — flatten-TP with no TP group.**
  `FusedMoEParallelConfig.make` non-EP branch
  (`vllm/model_executor/layers/fused_moe/config.py:1201` →
  `flatten_tp_across_dp_and_pcp`, `config.py:1096`) sets
  `tp_size = dp_size·tp_size = 8` and the MoE relies on
  `_maybe_reduce_final_output` →
  `tensor_model_parallel_all_reduce` (`runner/moe_runner.py:453,456`). In a pure-DP
  self-spec deployment `get_tp_group().world_size = 1`, so the all-reduce is a no-op
  and each rank emits a 1/8-TP-sharded MoE output. Proof:
  `probe_draft_tp_group.py` → `moe_tp_size=8 ... get_tp_group().world_size=1`.
  **Not yet fixed** (out of scope for the bug hunt): a genuine comm-free full
  replica needs the draft MoE built with `tp_size=ep_size=dp_size=1` (each rank a
  standalone single-GPU MoE holding unsharded experts), which `make` does not
  currently support for the DP-without-EP case. This is the remaining blocker to
  accept ~5.0.

---

## The real reason config A accepts 2.18 not 5.0 at DP=8

Not bf16-reduce associativity (phase 40 ruled that out: FP32-accum did nothing).
The "full replica" was never a replica: the draft was silently EP-sharded to
7–8/60 experts/rank (Bug A), and even forced to 60/60 it is TP-sharded 8-way with
no TP group to reduce it (Bug B). Either way the draft computes a structurally
wrong MoE output with the correct routing, the depth-0 draft token mismatches the
EP verify, and acceptance collapses with DP width (30/60 at DP=2 → accept 4.88;
7–8/60 at DP=8 → accept 2.18). The phase-40 main-model proxy was misleading because
`enable_expert_parallel=False` on the main model (single process per DP rank, no
flatten) IS a true replica, whereas the real draft path is not.

## Files
- Fix: `vllm/v1/spec_decode/llm_base_proposer.py` (`_create_draft_vllm_config`
  override + `set_current_vllm_config` around draft `get_model`).
- Probes: `scripts/probe_draft_vs_verify_state.py`,
  `scripts/probe_draft_config_discarded.py`, `scripts/probe_draft_tp_group.py`,
  `scripts/probe_moesum_slots.py`.
- Logs: `logs/draft_vs_verify_state_dp8/`, `logs/draft_config_discarded_dp8/`,
  `logs/draft_tp_group_dp8/`, `logs/draft_state_prefix_dp{2,8}/`.
- Data: `data/dp_sweep/` (A/D MoE dumps DP=1/2/4), `data/acc_A_dp8_K4_cg_postfix.json`.
