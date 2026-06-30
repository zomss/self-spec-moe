# Phase 38: decisive EP-isolation MEASUREMENT (which lever recovers accept at large EP)

**Source phase:** 37 (compile-consistency). Phase 37 fixed the per-step COMPILE
divergence (`VLLM_SELF_SPEC_COMPILE_CONSISTENT=1`) but found that at **DP=8 + EP=8
the comm-free full-replica draft still collapses to accept ~1.0 — present even
EAGER**, so it is a SEPARATE structural divergence: the comm-free replica sums its
experts LOCALLY vs the verify's EP all-to-all (an FP-reduction divergence that
worsens with EP width).

**Objective (no source changes):** before building either of two multi-week levers,
measure whether they would recover accept at large EP.
- **L1 — EP-path draft:** route the draft through the EP path (matching the verify's
  reduction order for resident experts) instead of the non-EP full replica.
- **L2 — verify-context-KV:** the draft attends to the verify's clean KV instead of
  its own drifted KV.

## Setup (fast proxy)
`Qwen/Qwen1.5-MoE-A2.7B` (qwen2_moe, 60 experts top-4, 24 layers, non-MLA),
**DP=8 -> EP=8**, forced-PCIe (`NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0
NCCL_IB_DISABLE=1`), greedy, **K=4**, **bf16** (no FP8 confound). Always
`VLLM_SELF_SPEC_DRAFT_FULL_CG=1` + `VLLM_SELF_SPEC_COMPILE_CONSISTENT=1` (compile
bug FIXED -> only remaining divergence is comm-free-vs-EP). draft_model self-spec,
draft = same model. `VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0`.

## Test 1 — which draft routing recovers accept (isolates L1)
| config | flags | meaning |
|---|---|---|
| A full-replica comm-free | `DRAFT_FULL_REPLICA=1 DRAFT_LOCAL_ROUTE=1` | the collapse (local sum of ALL experts) |
| B EP-shard comm-free | `DRAFT_FULL_REPLICA=0 DRAFT_LOCAL_ROUTE=1` | EP path + skip-A2A, resident experts only |
| C EP-full (upper bound) | `DRAFT_FULL_REPLICA=0 DRAFT_LOCAL_ROUTE=0` | draft does the real all-to-all = the EP verify |

Verdict: **B >> A** -> EP path avoids the full-replica FP divergence -> BUILD L1
(quantized EP-shard + hot cache). **B ≈ A (both ~1)** -> comm-free skip is itself
the wall -> L1 won't help. **C ≈ 5** confirms divergence is purely the comm-free skip.

## Test 2 — per-step or accumulated KV drift (isolates L2)
Per-position acceptance vector (`vllm:spec_decode_num_accepted_tokens_per_pos`):
pos-0 rate = accepted-at-depth-0 / num_drafts. If depth-0 is already ~rejected ->
per-step MoE divergence -> L2 won't help. If depth-0 is fine but deeper collapses
-> accumulated KV drift -> L2 would help. Cross-check K=1 vs K=4.

## Commands
```
bash research/38_ep_isolation/scripts/run.sh   # serial, DP=8, A/B/C K4 + A/C K1
```
Per-run JSON in `data/`, logs in `logs/`. Harness tears down its OWN workers
between runs (idle GPUs only).

## Output
`results_W7_ep_isolation.md` — Test 1 table, Test 2 per-position, the build verdict.
