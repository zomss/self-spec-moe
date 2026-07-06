# Phase 65 results — draft-path overhead optimization at 16k, 2-node EP16

Setup: identical to Phase 64 (h107+h106, DP16/EP16, ctx 16,384 distinct
per-request chat + on-dist prompts, max_model_len 20,480, MNB 8192,
gpu_mem 0.90, greedy, two-length slope 160/32, iters=2 warmup=1, harness
w7_2node.py). Self-spec W512 sinks=16 self-draft; b32 remains pool-blocked
(Phase 64: 2.3x over-pool preemption livelock) and was not run.

Stages (cumulative):

- **f1** = `VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1` — draft-path DP
  coordination on the gloo CPU group (zero GPU->CPU syncs in the propose
  path).
- **f12** = f1 + `VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1` — chain builds its
  FA metadata once, per-step updates are data-only in-place writes +
  max_seq_len scalar; persistent compaction gather buffer; cached
  slot-mapping dict.
- **f123** = f12 + the banked Phase-52 comm-free draft
  (`W7_DRAFT_QUANT=fp8 W7_DRAFT_FULL_REPLICA=1 W7_DRAFT_LOCAL_ROUTE=1`).

## Stage table — decode tok/s (accept_len), 16k

(filled by scripts/analyze.py)

TBD

## Fixed/marginal split (b8, cycle = F + K*D)

TBD

## Smoke validation (single-node DP4/EP4, 2k ctx, b4)

| stage | K | accept_len | note |
|---|---|---|---|
| f1 | 2 | 2.850 | window engaged (win_seq 541 of ~2045) |
| f12 | 4 | 4.365 | light-MD active on chain steps 2-3 |
| f1 | 4 | **4.365** | == f12 EXACTLY (greedy A/B: light-MD is inert) |
| f123 | 4 | 4.273 | fp8 replica loaded (use_ep=False), small accept cost |

fp8-replica pool cost at DP4/EP4 gpu_mem 0.85: 142,896 -> 77,616
tokens/rank (delta 65.3k tokens = 12.0 GiB) — smaller than the Phase-62
EP8 delta (18.7 GiB); the EP16 number is read from the f123 run logs.

## Phase-66 groundwork — capping / sharing the drafter's KV

Where the drafter's KV is sized: the draft model's 48 `Attention` layers
register in the static forward context; the runner's
`get_kv_cache_spec()` (vllm/v1/worker/gpu_model_runner.py:7679) collects
one spec per layer, and since draft layers have `sliding_window=None`
they return `FullAttentionSpec` (vllm/model_executor/layers/attention/
attention.py:581) — identical to the target's. All 96 layers therefore
land in ONE KV-cache group with shared block tables and a per-token page
cost of 192 KiB (the Phase-64 pool-halving discovery). There is no
"drafter pool" to cap: draft KV rides the same block allocation as
target KV, block for block.

A `VLLM_SELF_SPEC_DRAFT_KV_POOL_TOKENS` cap therefore needs the draft
layers in a SEPARATE kv-cache group with its own (smaller/wrapping)
allocation, i.e. the hybrid-KV-cache-manager path:

1. Draft `Attention.get_kv_cache_spec()` returns a windowed spec (e.g.
   `SlidingWindowSpec(sliding_window=cap)`) when the flag is set — the
   manager then allocates/frees draft blocks in a window automatically.
2. The propose path must consume the DRAFT group's block table + slot
   mappings (today it reuses the target-group `common_attn_metadata`;
   the per-layer `slot_mappings` dict argument already exists).
3. Sinks: `SlidingWindowSpec` frees the FIRST blocks too, killing the
   16 attention-sink tokens the Phase-62 window keeps. Needs either a
   sink-aware spec (keep first block + trailing window — the natural
   "StreamingLLM spec") or a measured accept at sinks=0.
4. `validate_same_kv_cache_group` (all draft layers in one group) still
   holds; scheduler-side hybrid-manager compatibility with the padded
   drafter batch is the main untested surface.

The full Phase-66 fix (true shared-KV self-draft: draft READS the
target's cache, writes nothing but its window tail) removes the second
allocation entirely and restores b32 residency (522k of 567k
tokens/rank).
