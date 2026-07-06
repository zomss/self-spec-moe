# Phase 64 — Window-KV self-draft END-TO-END serving throughput at 16k (2-node EP16)

Source phases: 62 (window-KV draft mechanism + 16k accept ablation), 59
(long-context methodology + same-cell no-spec/EAGLE references), 63 (fixed
2-node retry runner), 52 (harness + self-spec env stack).

## Objective

Phase 62 proved the accept side: the bf16 EP-routed self-draft with KV window
512 (sinks 16) keeps per-token accept r = 0.957 at 16k on Qwen3-30B-A3B.
This phase measures what that buys END-TO-END: serving decode tok/s at 16k on
the real 2-node fabric (h107+h106, DP16/EP16), window arms vs a same-session
no-spec denominator (and, optionally, a fresh EAGLE3 K=1 reference).

Economic model under test: at 16k the step is ~65% KV read + ~25% FFN + ~10%
comm; the windowed draft step should cost ~0.35x of a full step, verify
tolerates K=4 free (Phase 59: throughput flat K1-K4), so a K=4 cycle is
~4x0.35 + 1.0 = 2.4 step-equivalents per ~4.58 accepted tokens ->
predicted ~1.9x vs no-spec, training-free (EAGLE's measured band at this
cell: 1.3-1.8x).

## Setup (all arms)

Qwen3-30B-A3B, W7_NODES=2 W7_LOCAL_WORLD=8 (DP16/EP16), 8-rail fabric
(`NCCL_IB_HCA='^mlx5_8'`, gloo/bootstrap on enmlx0), W7_CTX_TOKENS=16384,
W7_MAX_MODEL_LEN=20480, W7_GPU_MEM=0.90, W7_MAX_NUM_BATCHED=8192, chat +
on-dist prompts (phase-57 bank), batches 8,32 (b64 excluded: over-pool,
Phase 59), iters=2 warmup=1, greedy, two-length decode slope (160/32).
Harness: `research/52_two_node_e2e/scripts/w7_2node.py` (unchanged).

Arms (one engine invocation each, ports 14200+, TRY_TO 1200s, retry once):

1. `nospec` — same-session denominator (Phase 59 env stack; ref 348.5/367.9).
2. `w512k2` — self-spec, VLLM_SELF_SPEC_DRAFT_KV_WINDOW=512, K=2.
3. `w512k4` — same, K=4 (the headline arm).
4. `w256k4` — WINDOW=256, K=4 (accept identical per Phase 62; does the
   smaller window buy measurable step time?).
5. `eagle_k1` (optional) — EAGLE3 `Tengyunw/qwen3_30b_moe_eagle3`, K=1.

Arms 2-4 use the EP-routed bf16 self-draft: env
`scripts/env_selfspec_2node.sh` (Phase-52 stack, quant REMOVED) with
W7_DRAFT_QUANT= W7_DRAFT_FULL_REPLICA=0 W7_DRAFT_LOCAL_ROUTE=0
W7_DRAFT_NODE_LOCAL=0; sinks default 16; W7_KV_WINDOW_DEBUG=1 for
engagement evidence. Arms 1/5 use
`research/57_large_ep_spec_strategy/scripts/env_eagle_2node.sh` read-only
(exactly the Phase 59 references' env).

## Commands

```bash
bash scripts/run_all.sh                  # all measured points, sequential
bash scripts/run_arm.sh eagle_k1 8,32 4 1500   # optional arm 5 (dropped)
bash scripts/run_trace16k.sh 2 8 60      # rank-0 torch trace, W512 K=2 b8
.venv/bin/python scripts/analyze.py      # table from data/*.json
.venv/bin/python scripts/analyze_trace16k.py data/trace_w512k2_b8/
```

Runner: clone of the FIXED Phase 63 runner (bounded wait on the h106 ssh —
orphaned engine children hold the pipe and wedge retry loops), plus the
Phase-62 foreign-GPU guard extended to BOTH nodes.

## Decision criteria / readout

- Headline: W512-K4 at b32 vs same-session no-spec — > 1.0x? near the ~1.9x
  model? vs EAGLE band 1.3-1.8x?
- Sanity: accept_len must reproduce Phase 62 (~4.58 at K=4/W512, ~2.87 at
  K=2 given r=0.957; W256 == W512). Deviation > 0.2 => flag (fabric must not
  change accept).
- KV-capacity watch item (identified pre-run): the draft_model path
  DUPLICATES per-token KV (Phase 62 logs: 40.59 GiB pool / 221,664 tokens =
  192 KiB/token = 2x the model's 96 KiB). At EP16 the self-spec pool is
  expected ~half of no-spec's 567k tokens/rank, so b32 (32x16.6k = 530k
  tokens/rank) may not be KV-resident for arms 2-4 -> scheduler waves
  (Phase 62 C/D pattern). Read "GPU KV cache size" per arm; report waves.
- Systems watch item: if W512-K4 badly underperforms the model at b32 with
  accept right, name the dominant draft-path overhead (harness has no
  profiler hook; report from logs).

Expected next artifact: `results_window_e2e.md` with the tok/s (accept) x
{b8,b32} table, speedups vs no-spec, and the headline verdict.

## Outcome (2026-07-06)

See `results_window_e2e.md`. Headline NEGATIVE: W512 self-draft = 0.34x
(K=2 b8) / 0.38x (K=4 b8) / 0.51x (K=4 b12) of same-session no-spec at 16k
EP16, with accept exactly reproducing Phase 62 (2.869 / 4.673-4.695) and
the window engaged (draft reads ~530/16.4k KV tokens). The pre-run KV
watch item confirmed: draft KV duplication (192 KiB/token) halves the pool
to 228k tokens/rank, so b32 is 2.3x over-pool -> preemption livelock (no
measurable point). Mid-phase scope re-cut with the coordinator: W256 +
EAGLE arms dropped; added b12 resident points and a rank-0 torch trace
(W512 K=2 b8) that names the draft-path overheads: per-cycle `.item()`
DP-coord sync storm (~32 ms/step), per-draft-forward cross-node EP
AllGather+ReduceScatter (~26 ms/forward, 45%+21% of window CUDA time),
eager per-layer chain dispatch (~22 ms/step CPU). Phase 65 = draft-path
optimization: shared-KV drafter, sync elimination, draft comm locality.
