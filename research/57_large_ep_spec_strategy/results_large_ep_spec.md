# Phase 57 results — the optimal spec-decode strategy at large EP, on REAL 2-node fabric

**Thesis (from README).** On MoE-EP the verify is one all-to-all over EVERY
drafted token, so exposed verify comm scales with draft **VOLUME** (chain
length / tree size), not with accepted tokens. The objective at large EP is
therefore **accept-per-verified-token**, and the optimum is **minimal draft
volume** — a short chain (K\* small, often 1), inverting single-GPU wide-tree
practice. As EP scale / comm-fraction f grows, K\* shrinks.

**What is new here vs Phase 33/50.** Every prior World B number was
single-node forced-PCIe EMULATION (Phase 33 used a local-routing stand-in for
EAGLE acceptance; Phase 50 used a real EAGLE3 head but on one node with f
emulated by an injected all-to-all delay). This phase measures the same law
and K\* with the REAL trained EAGLE3 head on the genuine 2-node inter-node
fabric (h107+h106, EP16), and compares EP8 (1-node) vs EP16 (2-node) to show
K\*(EP).

- Real EAGLE3 head: `Tengyunw/qwen3_30b_moe_eagle3` (base `Qwen/Qwen3-30B-A3B`).
- Harness: `research/52_two_node_e2e/scripts/w7_2node.py` with
  `W7_SPEC_METHOD=eagle3`; clean EAGLE env
  `research/57_large_ep_spec_strategy/scripts/env_eagle_2node.sh` (NO
  `VLLM_SELF_SPEC_*` stack, cf. Phase 50 STACK=0).
- tok/s = two-length decode slope (OUTLEN 160 vs SHORTLEN 32); accept_len from
  `vllm:spec_decode_num_accepted_tokens / num_drafts + 1`. Greedy, ignore_eos,
  off-distribution synthetic prompts (accept is depressed vs on-distribution,
  same caveat as Phase 50).

## Fixes required to run EAGLE on this self-spec branch

1. **Harness env guard** (`w7_2node.py`): the worker set
   `VLLM_SELF_SPEC_DRAFT_LOCAL_ROUTE / NODE_LOCAL / FULL_REPLICA / LOCAL_ROUTE`
   unconditionally in spec mode. Those are self-spec-only; for a real EAGLE
   head they are wrong. Now guarded by `spec_method == "draft_model"`.
2. **EP-propagation bug** (`vllm/config/speculative.py`): the self-spec fork's
   `create_draft_parallel_config` propagated `enable_expert_parallel=True` from
   the MoE target onto the draft. A dense EAGLE head has 0 experts, so
   `ModelConfig._verify_with_expert_parallelism` rejected it
   ("Number of experts ... must be greater than 0"). Fixed to propagate EP only
   when the draft is itself MoE (`draft_is_moe=self.draft_model_config.is_moe`).
   Self-spec (draft == the MoE target) is unchanged.

## STAGE 1 — smoke (de-risk EAGLE on this HEAD): PASS

1-node DP8/EP8, Qwen3-30B + real EAGLE3, K=2, b8, clean EAGLE env, W7_GPU_MEM
0.90. EAGLE3 loads and routes; **accept_len = 1.716** — sane, and consistent
with Phase 50's ~1.60-1.67 (slightly higher here; on-distribution the head is
~2.3-2.5). tok/s at b8/1-iter is noisy (single small batch); the trustworthy
throughput curve is the STAGE 2/3 sweep below.

## STAGE 2 — real 2-node EAGLE3 K-sweep (EP16)

_(pending)_

## STAGE 3 — K\*(f): EP8 vs EP16

_(pending)_
