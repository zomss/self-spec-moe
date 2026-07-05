# Phase 59 results — does spec-decode's serving-batch speedup rise with context on multi-node MoE-EP?

_DRAFT — being filled as runs complete. Source: Phase 57 (short-ctx serving-batch
loss) + Phase 58 (MagicDec long-context hypothesis)._

## Question

Phase 57 measured, at SHORT context (~2k), that EAGLE3 spec decoding on Qwen3-30B
at 2-node EP16 is a **low-batch win only** (b8 1.40x, b32 1.30x, b64 **0.95x** —
a serving-batch loss). MagicDec (2408.11049) predicts decode stays KV/memory-
bound at ANY batch for LONG context, so spec decoding should help even at serving
batch, and the speedup should rise with context. **Does our short-context
serving-batch loss flip to a win as context grows, on real multi-node MoE-EP?**

## Memory / context envelope (worked out first)

- Qwen3-30B-A3B: `num_hidden_layers=48`, `num_key_value_heads=4`, `head_dim=128`,
  bf16 -> **KV = 2*48*4*128*2 B = 96 KiB / token**.
- Per-request full KV: 2k -> 0.19 GiB, 16k -> 1.5 GiB, 32k -> 3.0 GiB.
- **Context ceiling is `max_position_embeddings=40960` (native ~40k), NOT
  memory.** So LONG = **32k** (inside 40960, the MagicDec regime); going to 64k
  would require RoPE scaling (untrained -> off-distribution) and is excluded.
- Memory is generous: at EP16 (attention-DP16) each rank holds KV for its
  `global_batch/16` requests. Measured at 32k: engine steady-state ~75.5/80 GiB
  per rank (6.26 GiB weights + pre-allocated KV pool), ~4.5 GiB headroom. At b64
  each rank holds only 4 req x 3.0 = 12 GiB of active KV vs a ~65 GiB block pool
  -> **b64 fits at 32k with large margin; no OOM.**
- **Design: FIXED batch set {8, 32, 64} across all three contexts** (option (a),
  apples-to-apples). b64 is the serving-batch point that LOST at short context
  (0.95x) — the decisive cell.
- `W7_MAX_NUM_BATCHED=4096` (bounds fused-MoE workspace; chunked prefill handles
  the 32k prompt in 4096-tok chunks). `W7_GPU_MEM=0.90`, CUDA graphs on.
  Long-context runs use `W7_ITERS=2 W7_WARMUP=1` (warmup caches the big prefill;
  the two-length slope then isolates decode over the full KV).

## Context x {no-spec, EAGLE K1, K2} tok/s and speedup

**SHORT 2k** is reused verbatim from Phase 57 (chat, on-dist, EP16 — same harness,
same prompts, `W7_CTX_TOKENS` unset = byte-identical path).

<!-- FILL: 16k and 32k rows -->

## Headline

<!-- FILL -->

## Caveats

<!-- FILL -->
