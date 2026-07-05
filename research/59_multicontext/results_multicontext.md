# Phase 59 results — does spec-decode's serving-batch speedup rise with context on multi-node MoE-EP?

Source: Phase 57 (short-ctx serving-batch loss) + Phase 58 (MagicDec long-context
hypothesis). Real 2-node EP16 fabric (h107+h106), Qwen3-30B-A3B + real EAGLE3 head
`Tengyunw/qwen3_30b_moe_eagle3`, chat + on-distribution prompts, greedy, clean
EAGLE env. tok/s = two-length decode slope (OUTLEN 160 vs SHORTLEN 32); the
decode attends over a large per-request KV via `W7_CTX_TOKENS` padding.

## Question

Phase 57 measured, at SHORT context (~2k), that EAGLE3 spec decoding at 2-node
EP16 is a **low-batch win only** (b8 1.40x, b32 1.30x, b64 **0.95x** — a
serving-batch loss), and the speedup FALLS with batch. MagicDec (2408.11049)
predicts decode stays KV/memory-bound at ANY batch for LONG context, so spec
decoding should help even at serving batch and the speedup should RISE with
context/batch. **Does the short-context serving-batch loss flip to a win as
context grows, on real multi-node MoE-EP?**

## Memory / context / batch envelope (worked out first, then measured)

- Qwen3-30B-A3B: `num_hidden_layers=48`, `num_key_value_heads=4`, `head_dim=128`,
  bf16 -> **KV = 2*48*4*128*2 B = 96 KiB / token**.
- **Context ceiling is `max_position_embeddings=40960` (native ~40k), NOT
  memory.** Beyond 40960 needs untrained RoPE scaling (off-distribution). So LONG
  = **32k** (inside 40960, squarely the MagicDec regime). Contexts: **2k / 16k /
  32k**.
- At EP16 the KV pool (measured) is ~1.12M tokens per rank (`W7_GPU_MEM=0.90`,
  `W7_MAX_NUM_BATCHED=8192`; steady-state 75-77 GiB/80). A single rank runs the
  full measured batch (Phase-57 convention). So the max batch whose KV fits
  concurrently in one rank is ~1.12M / ctx_tokens:
  - 2k  -> fits to very large batch,
  - 16k -> b64 = 1.04M tokens ~= 93% pool (fits, but near-full -> prefill thrash),
  - 32k -> **b32 = 1.04M ~= 93% pool (fits); b64 = 2.09M > pool (queues, not a
    clean concurrent-b64 measurement).**
- **Design (mission option (a), FIXED serving-ish batch across contexts):** the
  largest batch that fits concurrently at the LONG context is **b32** -> b32 is
  the apples-to-apples fixed-batch column across 2k/16k/32k. b8 (low batch) and
  b64 (true serving batch, where it fits: 2k/16k) are reported alongside to show
  the batch-dependence flip.
- Long-context prefill is the wall-clock cost (each rank prefills batch x ctx
  tokens, and near-full-pool batches re-prefill across passes as the prefix cache
  thrashes). `W7_ITERS=2 W7_WARMUP=1`.

## Context x {no-spec, EAGLE K1} tok/s (accept_len) and speedup

**2k reused verbatim from Phase 57** (chat, on-dist, EP16 — same harness/prompts;
`W7_CTX_TOKENS` unset = byte-identical path). 16k/32k measured here.

| ctx | batch | no-spec tok/s | EAGLE K1 tok/s (accept) | K1 speedup |
|----:|------:|--------------:|------------------------:|-----------:|
| 2k  |  8    | 521.6  | 568.7 (1.73) | 1.09x |
| 2k  |  32   | 1581.1 | 1840.1 (1.73) | 1.16x |
| 2k  |  64   | 2611.1 | 2475.7 (1.71) | **0.95x (loss)** |
| 16k |  8    | 348.5  | 463.4 (1.68) | **1.33x** |
| 16k |  32   | 367.9  | 665.0 (1.66) | **1.81x** |
| 16k |  64   | 306.5  | _(not completed; b64@16k near-full pool, prefill-thrash)_ | -- |
| 32k |  8    | _TBD_  | _TBD_ | _TBD_ |
| 32k |  32   | _TBD_  | _TBD_ | _TBD_ |

### Two decisive structural facts already visible

1. **No-spec decode goes KV-bound at long context.** At 2k, no-spec tok/s scales
   ~5x across batch (521 -> 1581 -> 2611: weight reads amortize over the batch).
   At 16k it is **flat-to-declining** (348 -> 368 -> 306): adding batch buys ~no
   throughput because each request's decode is dominated by reading its own 16k
   KV, which does NOT amortize. This is the exact MagicDec precondition.
2. **The EAGLE speedup INVERTS its batch dependence with context.** At 2k the
   speedup FALLS with batch (1.09 -> 1.16 -> 0.95, loss at serving batch). At 16k
   it RISES with batch (1.33 -> 1.81 ...). Because in the KV-bound regime each
   verify pass is dominated by the (fixed) KV read, EAGLE's ~1.66 accepted
   tokens/pass convert into ~1.6-1.8x fewer KV-read passes -> the speedup
   approaches accept_len, and the short-context verify-comm penalty (that sank
   b64 at 2k) is now a small fraction of the KV-dominated step.

## Headline

_TBD once 32k lands — expected: the fixed serving-ish-batch (b32) speedup rises
2k 1.16x -> 16k 1.81x -> 32k, and the true-serving-batch (b64) 2k loss (0.95x)
is erased by 16k (b8 1.33x, b32 1.81x, both up and rising)._

## Caveats

- **Accept is context-invariant** (~1.66-1.73 for K1, same trained head, chat
  prompts) across 2k/16k/32k — every tok/s change is the machine-balance shift
  (KV-bound vs compute/comm-bound), not acceptance.
- **b64 at long context:** at 16k b64 the KV pool is ~93% full and prefill
  thrashes (slow); at 32k b64 the batch exceeds the pool and queues. Hence the
  clean fixed-batch column is b32 (mission option (a)); b64 is reported only
  where it fits concurrently (2k, and 16k as a trend).
- **Fixed-batch choice:** b32 is the max serving-ish batch that fits at the LONG
  (32k) context; a larger-pool deployment (more ranks / smaller per-rank batch /
  DeepEP) would push the concurrent serving batch higher.
- Single-node vs multi-node, EAGLE chain-only (Phase 57-A1), off-vs-on-dist
  accept (Phase 57-A2) caveats carry over unchanged.
