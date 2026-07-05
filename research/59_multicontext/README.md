# Phase 59 — Does the serving-batch speedup of speculative decoding on multi-node MoE-EP RISE with context length?

## Source (where this comes from)

- **Phase 57** (`research/57_large_ep_spec_strategy/results_large_ep_spec.md`)
  measured, on the REAL 2-node EP16 fabric, that EAGLE3 spec decoding on
  Qwen3-30B is a **low-batch win only**: at SHORT context (~2k), on-distribution
  chat prompts, EAGLE-over-no-spec speedup = **1.40x (b8) / 1.30x (b32) /
  0.95x (b64)** — a LOSS at serving batch. Mechanism: on MoE-EP the verify is an
  all-to-all over EVERY committed token; at serving batch the step is
  bandwidth-bound and the (k+1)-irreducible verify-comm penalty exceeds the
  shrinking compute saving.
- **Phase 58** (`research/58_literature_restart/directions.md`) situated that
  negative against the literature and found it is **regime-specific**: measured
  only at short context, exposed-a2a. The literature names a serving-batch WIN
  regime we never tested — long context.

## The MagicDec hypothesis (what this phase tests)

**MagicDec (ICLR 2025, arXiv:2408.11049):** KV-cache load scales with
`batch x seqlen` and does NOT amortize across the batch the way weight reads do.
So at LONG context decode stays **memory/KV-bound at ANY batch**, and
speculative decoding helps even at large batch — speedup *increases* with
batch/context. This is the exact opposite of our short-context serving-batch
loss.

**Question:** on real multi-node MoE-EP (EP16, exposed inter-node verify a2a),
does the short-context serving-batch LOSS become a WIN as context grows? Is
there a crossover to >1.0x, and at what context?

## Setup

- Harness `research/52_two_node_e2e/scripts/w7_2node.py` (two-length decode-slope
  tok/s). Phase-59 addition: **`W7_CTX_TOKENS`** pads each on-dist prompt to
  ~that many tokens with rotated natural-text filler + a per-request doc marker,
  INSIDE the user turn (chat template applied after), so decode attends over a
  large, DISTINCT per-request KV (no shared cached prefix). Default unset ->
  byte-identical.
- Real EAGLE3 head `Tengyunw/qwen3_30b_moe_eagle3`, target `Qwen/Qwen3-30B-A3B`.
- Clean EAGLE env `research/57_large_ep_spec_strategy/scripts/env_eagle_2node.sh`,
  on-dist prompts `.../data/prompts_ondist.txt` (80), `W7_CHAT=1`.

## Context ceiling (envelope)

Qwen3-30B-A3B has `max_position_embeddings=40960` (native ~40k). KV per token =
2*48*4*128*2 B = 96 KiB. At 32k a request holds 3.0 GiB of full KV; at DP16 b64
each rank holds only 4 req x 3.0 = 12 GiB, so **memory is not the binding
constraint — the native 40960 context ceiling is.** LONG is therefore set to
**32k** (well inside 40960, squarely the MagicDec regime). Contexts tested:
**2k (SHORT, reused from Phase 57), 16k (MIDDLE), 32k (LONG)**.

## Commands

Run each stage as a SINGLE detached job (`setsid`, own process group) — NOT via a
nesting orchestrator: a killed parent orphans the retry loop, which respawns
engines. `MNB=8192`. `TRY_TO` for EAGLE must EXCEED the healthy long-ctx run time
(~25 min at 16k b64), else a healthy run is false-wedge-killed and retried forever;
real wedges still fail fast via `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=120`.

```bash
# no-spec baseline (CTX MAXLEN BATCHES PORT [ITERS WARMUP MNB])
setsid bash scripts/run_nospec_ctx.sh 16384 17408 8,32,64 13700 2 1 8192 &
setsid bash scripts/run_nospec_ctx.sh 32768 33792 8,32    13710 2 1 8192 &

# EAGLE3 K1 (CTX MAXLEN BATCHES "KLIST" PORT ITERS WARMUP MNB TRY_TO RETRIES)
setsid bash scripts/run_eagle_ctx.sh 16384 17408 8,32,64 "1" 13820 2 1 8192 3600 3 &
setsid bash scripts/run_eagle_ctx.sh 32768 33792 8,32    "1" 13950 2 1 8192 3600 2 &
```

**Batch envelope:** each rank runs the full batch; the one-rank KV pool is ~1.0-1.1M
tokens. Batches whose KV (batch x ctx) approaches the pool re-prefill-thrash under
the two-length method (slow / noisy): 32k is clean only to ~b8-b16; 16k to ~b32.
Use b32 as the fixed serving-ish column (clean at 2k/16k). `scripts/analyze.py`
assembles the ctx x mode speedup table.

Results and the full memory/batch envelope: `results_multicontext.md`.
