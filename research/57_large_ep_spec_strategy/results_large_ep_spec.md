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
3. **Intermittent DP16 EAGLE deadlock** (infra, not a code bug we own). Stock
   EAGLE spec decode wedges the first `sample_tokens`/`execute_model` collective
   on the 2-node DP16 fabric where World A (self-spec) runs fine — because EAGLE
   lacks the self-spec DP-coordination stack. It is INTERMITTENT (a given
   K/batch sometimes completes, sometimes hangs) and independent of cudagraph vs
   eager (both wedge). Two partial mitigations + one real one:
   - EAGLE auto-enables async scheduling (self-spec `draft_model` runs it OFF);
     its batch-queue path wedges more readily. Force `async_scheduling=False`
     (env-gated `W7_ASYNC_SCHED`, unset -> vLLM default, self-spec unchanged).
   - `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=120` so a wedge fails fast.
   - The real mitigation: `run_ksweep.sh` runs ONE K per engine invocation
     wrapped in a hard `timeout`, force-kills wedged NCCL workers by GPU pid on
     both nodes, and RETRIES. Every K/batch point below is a completed
     (non-wedged) run. This DP16-EAGLE fragility is itself a finding: the
     conventional (non-self-spec) drafter is not robust at multi-node EP on this
     stack, independent of the throughput argument.

## STAGE 3 arm A — 1-node DP8/EP8 (lower f), CG, the K-sweep

Complete, all K first-try (single-node DP8 is robust). tok/s / accept_len:

| K | b8 | b32 | b64 |
|---:|---:|---:|---:|
| 1 | 694 /1.53 | 2054 /1.49 | **2783** /1.46 |
| 2 | **784** /1.72 | **2163** /1.68 | 2393 /1.60 |
| 3 | 725 /1.77 | 1843 /1.75 | 1863 /1.65 |
| 4 | 672 /1.84 | 1685 /1.78 | 1527 /1.67 |
| 6 | 607 /1.84 | 1335 /1.78 | 1249 /1.67 |
| 8 | 559 /1.84 | 1188 /1.78 | 1055 /1.67 |

- **accept saturates** by K≈4 (b8 1.84, b32 1.78, b64 1.67); every draft token
  past that is pure volume waste. Higher accept than Phase 50's 1.67 at small
  batch (b8 1.84) — batch/prompt-set dependent; still off-distribution synthetic.
- **K\* at EP8: b8/b32 → K\*=2, b64 → K\*=1.** Even single-node, the largest
  batch already prefers the shortest chain: at b64 tok/s falls monotonically
  1→8 (2783→1055, K8 = 0.38× K1). The wide-tree-is-free intuition already breaks
  at b64 EP8; the thesis predicts EP16 pushes K\* to 1 at smaller batch too.

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
