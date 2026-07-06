# Phase 65 — kill the per-draft-step overhead of the self-spec propose path

Source phase: `research/64_window_e2e` (window-KV self-draft E2E at 16k,
2-node DP16/EP16 = LOSS 0.34-0.51x; accept exactly as predicted, failure is
pure propose-path systems cost).

## Objective

Phase 64 attribution per draft cycle (K=2 b8: 175 ms; K=4 b8: 252 ms;
verify healthy at 20.7-24.7 ms): ~38.6 ms marginal per draft step +
~75-98 ms fixed per cycle, against a ~7 ms economic budget. Trace-named
causes, fixed in this priority order (measuring after each):

1. **Fix 1 — DP-coordination `.item()` sync storm (~31.6 ms/step traced).**
   The propose path coordinates across DP twice per cycle (step-0 + chain)
   via an NCCL all_reduce whose `.item()` readbacks cudaStreamSynchronize —
   the chain coordination drains the entire just-enqueued step-0 draft
   forward. Fix: `VLLM_SELF_SPEC_DRAFT_DP_COORD_CPU=1` routes the draft
   path's `coordinate_batch_across_dp` over the DP CPU (gloo) group: same
   result tensor, all readbacks CPU-side, zero GPU syncs in the propose
   path. (Why the Phase-55 flags "relocated the wait": AMORTIZE memoizes
   repeated shapes but the chain already coordinates once per cycle — the
   first NCCL coordination still drains the stream; SKIP requires a
   comm-free draft since EP collectives need DP-agreed padded sizes.)
2. **Fix 2 — chain dispatch CPU (~22 ms/step traced).**
   `VLLM_SELF_SPEC_DRAFT_CHAIN_LIGHT_MD=1`: (a) window compaction reuses a
   persistent gather-index buffer (no per-step [bs, n_cols] realloc);
   (b) the PIECEWISE/eager chain builds its FlashAttentionMetadata ONCE per
   chain and per-step updates only the data (in-place buffer writes) +
   the max_seq_len scalar; the per-layer slot-mapping dict is cached.
   FULL-CG chain steps for FA3 remain blocked by the Phase-35 finding (FA3
   decode kernel freezes host-side work distribution at capture; accept
   collapses) — documented, not attempted.
3. **Fix 3 — draft comm (~25.7 ms NCCL per draft forward).**
   Switch the draft to the banked Phase-52 comm-free path
   (`W7_DRAFT_QUANT=fp8 W7_DRAFT_FULL_REPLICA=1 W7_DRAFT_LOCAL_ROUTE=1`);
   accept WITH the window already measured (4.547 @K4 W512, Phase 62 arm C).
   Known blocker at 16k: the fp8 replica costs ~19 GiB -> the (already
   halved) drafter KV pool shrinks further; measure at the largest resident
   batch and document what the Phase-66 KV cap needs.

## Protocol (identical to Phase 64)

16k (`W7_CTX_TOKENS=16384 W7_MAX_MODEL_LEN=20480`), chat + on-dist prompts,
`W7_GPU_MEM=0.90`, MNB 8192, batches 8,12 (b32 remains pool-blocked:
2.3x over-pool preemption livelock — do not run), iters=2 warmup=1,
two-length slope 160/32, 8-rail env (`scripts/env_selfspec_2node.sh`).
W512 K=4 (headline) and K=2 (for the fixed-vs-marginal split via cycle
arithmetic: cycle_ms = decode_s/(128/accept_len), solve cycle = F + K*D).

References (Phase 64): no-spec 386.8 (b8) / 395.7 (b12); before-opt W512:
K4 148.2 (b8) / 202.9 (b12), K2 131.1 (b8). Accept must stay 4.5-4.7 @K4
(2.87 @K2) and the `[kv-window]` line must show win_seq ~530-544.

## Commands

```bash
# smoke (single-node DP4, ~2k ctx, b4):
scripts/smoke_dp2.sh {base|f1|f12|f123} [K]
# 16k 2-node stage measurement:
scripts/run_arm.sh w512k4 f1 8,12
scripts/run_arm.sh w512k2 f1 8
```

## Decision criteria

Success bars: Fix 1 alone: draft span 98.6 -> ~65 ms/step-ish; 1+2:
marginal draft step <= ~15 ms; 1+2+3: <= ~10 ms and E2E >= 1.3-1.6x at b12.
Accept drop below 4.5 @K4 = correctness regression -> investigate.

## Expected next artifact

`results_overhead_opt.md` with the stage-by-stage table (draft-step
marginal ms, fixed ms/cycle, E2E tok/s + speedup, accept) + Phase-66
groundwork notes (KV sharing/cap).
