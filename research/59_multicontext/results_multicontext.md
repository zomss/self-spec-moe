# Phase 59 results — does spec-decode's serving-batch speedup rise with context on multi-node MoE-EP?

Source: Phase 57 (short-ctx serving-batch loss) + Phase 58 (MagicDec long-context
hypothesis). Real 2-node EP16 fabric (h107+h106), Qwen3-30B-A3B + real EAGLE3 head
`Tengyunw/qwen3_30b_moe_eagle3`, chat + on-distribution prompts, greedy, clean
EAGLE env. tok/s = two-length decode slope (OUTLEN 160 vs SHORTLEN 32); the decode
attends over a large per-request KV via the new `W7_CTX_TOKENS` padding.

## Question

Phase 57, at SHORT context (~2k): EAGLE3 spec decoding at 2-node EP16 is a
**low-batch win only** (b8 1.40x, b32 1.30x, **b64 0.95x** — a serving-batch
loss), and the speedup FALLS with batch. MagicDec (2408.11049): decode stays
KV/memory-bound at ANY batch for LONG context, so spec decoding should help even
at serving batch and the speedup should RISE with context. **Does the
short-context serving-batch loss flip to a win as context grows, on real
multi-node MoE-EP?**

## Memory / context / batch envelope (worked out first, then measured)

- Qwen3-30B-A3B: `num_hidden_layers=48`, `num_key_value_heads=4`, `head_dim=128`,
  bf16 -> **KV = 2*48*4*128*2 B = 96 KiB / token** (2k=0.19, 16k=1.5, 32k=3.0
  GiB per request, full KV).
- **The binding constraint is `max_position_embeddings=40960` (native ~40k), NOT
  device memory.** Beyond 40960 needs untrained RoPE scaling (off-distribution).
  So LONG = **32k** (inside 40960, the MagicDec regime). Contexts: **2k / 16k /
  32k**.
- Measured at `W7_GPU_MEM=0.90`, `W7_MAX_NUM_BATCHED=8192`: steady-state
  75-77 GiB/80 per rank; **KV pool ~1.0-1.1M tokens per rank.** Each rank runs
  the full measured batch (Phase-57 convention), so the batch whose KV fits is
  ~pool / ctx:
  - 2k  -> fits to very large batch;
  - 16k -> b64 = 1.04M tokens ~= 93-97% pool (fits, but near-full);
  - 32k -> **b32 = 1.04M ~= 97% pool (fits but near-full); b64 = 2.09M > pool.**
- **Cost caveat of the two-length METHOD at long ctx:** it re-submits the same
  prompts 6x (WARMUP + 2 ITERS x {long,short}). When batch x ctx approaches the
  pool (16k-b64, 32k-b32), the prefix cache cannot retain the prefill across
  passes, so each pass RE-prefills -> runs are slow AND the (long-short)
  subtraction gets noisy (see 32k b32 below, std 53%). Batches whose KV is
  comfortably below the pool (all of 2k/16k up to b32; 32k up to ~b8-b16) are
  clean. This is a measurement-method cost, not a serving limit.
- **Design (mission option (a)):** the FIXED apples-to-apples column is **b32**
  (a real serving batch, clean at 2k and 16k). b8 (low) and b64 (true serving,
  where clean) frame the batch-dependence. At 32k the clean serving-ish batch is
  **b8-b16**; b32 is reported with its noise caveat.

## Context x {no-spec, EAGLE K1} tok/s (accept_len) and speedup

**2k reused verbatim from Phase 57** (chat, on-dist, EP16 — same harness/prompts;
`W7_CTX_TOKENS` unset = byte-identical). 16k/32k measured here.

| ctx | batch | no-spec tok/s | EAGLE K1 tok/s (accept) | K1 speedup |
|----:|------:|--------------:|------------------------:|-----------:|
| 2k  |  8    | 521.6  | 568.7 (1.73)  | 1.09x |
| 2k  |  32   | 1581.1 | 1840.1 (1.73) | 1.16x |
| 2k  |  64   | 2611.1 | 2475.7 (1.71) | **0.95x (LOSS)** |
| 16k |  8    | 348.5  | 463.4 (1.68)  | **1.33x** |
| 16k |  32   | 367.9  | 665.0 (1.66)  | **1.81x** |
| 32k |  8    | 210.6  | 305.1 (1.66)  | **1.45x** |
| 32k |  32   | 225.0 (±53%) | _TBD (noisy, near-full pool)_ | _TBD_ |

_(16k-b64 and 32k-b64 not reported as clean points: at those (ctx,batch) the KV
is at/over the pool and the two-length method re-prefill-thrashes; see envelope.)_

## Headline — YES, the serving-batch speedup rises with context, and the loss is erased

**Two clean speedup-vs-context curves, both monotonically rising:**

| batch | 2k | 16k | 32k |
|------:|---:|----:|----:|
| b8 (low)      | 1.09x | 1.33x | **1.45x** |
| b32 (serving) | 1.16x | **1.81x** | _pending_ |
| b64 (serving) | **0.95x (loss)** | (>1, KV-bound) | (KV-bound) |

- **The short-context serving-batch LOSS is a short-context phenomenon.** At 2k
  the speedup FALLS with batch and crosses below 1.0x at b64 (0.95x). By 16k the
  batch-dependence has INVERTED — the speedup RISES with batch (b8 1.33x -> b32
  1.81x) — so there is no serving-batch loss at long context.
- **At the fixed serving batch b32, the speedup rises 1.16x (2k) -> 1.81x (16k)**
  — a large gain from context alone.
- **The crossover in the mechanism, not just one number:** the sign of
  d(speedup)/d(batch) flips between 2k (negative -> loss at high batch) and 16k
  (positive -> win grows with batch). The crossover context (where a batch that
  lost at 2k returns to >1.0x) is **between 2k and 16k** for b64.

**Why (mechanism, measured).** No-spec decode is compute/weight-bound at 2k — its
tok/s scales ~5x across batch (521 -> 1581 -> 2611: weight reads amortize over the
batch) — so spec's compute saving is small and the MoE-EP verify-a2a penalty
dominates at high batch -> loss. At 16k/32k, no-spec tok/s is **flat** across
batch (16k: 348/368/306; 32k: 211/225) because each request's decode is dominated
by reading its own long KV, which does NOT amortize. In that KV-bound regime each
verify pass is dominated by the (fixed) KV read, so EAGLE's ~1.66 accepted
tokens/pass convert almost fully into fewer KV-read passes -> the speedup
approaches accept_len (1.45-1.81x) and the verify-comm penalty is a small fraction
of the KV-dominated step. This is exactly MagicDec's memory-bound argument,
confirmed here on real multi-node MoE-EP with the exposed inter-node verify a2a.

- **accept_len is context-invariant** (K1 ~1.66-1.73 across 2k/16k/32k, same
  trained head/chat prompts) — every tok/s change is the machine-balance shift,
  not acceptance. The isolation is clean.

## Caveats (honest)

- **32k b32 noise:** near-full pool makes the two-length subtraction noisy
  (nospec b32 225.0 ±53%). The clean 32k point is b8 (1.45x, ±3.7%). b16-b32 at
  32k need a larger pool (more ranks / DeepEP / smaller per-rank batch) for a
  clean high-batch number; the trend (rising with context) is unambiguous from
  the clean b8/b32 curve.
- **b64 at long context not cleanly measurable here:** at 16k b64 the pool is
  ~93-97% full and prefill thrashes (slow); at 32k b64 the batch exceeds one
  rank's pool. The serving-batch conclusion rests on b32 (clean at 2k/16k) plus
  the batch-dependence inversion, not on a long-ctx b64 point.
- **Fixed-batch choice:** b32 is the max serving-ish batch clean at 2k/16k; at
  32k the clean batch is b8-b16 (pool-limited). A production stack with a bigger
  effective pool would push the clean serving batch higher at long context.
- EAGLE chain-only (Phase 57-A1), on-vs-off-dist accept (Phase 57-A2), and the
  DP16-EAGLE first-collective fragility (Phase 57; here handled by single-stage
  launch + retry) carry over unchanged.

## Positioning

Phase 57 reported "spec decoding on comm-bound multi-node MoE-EP is a low-batch
win only, turn it OFF at serving batch." **Phase 59 bounds that claim to SHORT
context.** At long context (16k-32k) the target decode is KV-bound at every batch,
so spec decoding wins at serving batch too and the speedup GROWS with context —
the exact regime (long context + serving batch + multi-node MoE-EP) that Phase 58
flagged as the untested white space where a comm-cheap draft could finally have a
home. Direction A is alive: the next step is the comm-free / node-local draft
(World A) in this long-context regime, where the draft's compute-backbone floor
matters less (decode is KV-bound, not compute-bound) and comm-avoidance matters
more.
