# Phase 63 results — MLA does NOT remove the long-context KV-bound regime

**Question.** Phase 59 showed GQA Qwen3-30B (96 KB/token KV) collapses ~4.3x at
b32 going 2k -> 16k and goes batch-flat (KV-bound at any batch — the MagicDec
regime where long-context spec decode wins at serving batch). Does DeepSeek-V2
236B (MLA, ~70 KB/token latent) escape that regime at 16k — staying
compute/comm-bound, batch-scaling, with only a ~1.2-1.5x drop?

**Answer: NO.** At 16k the MLA model is batch-flat from b8 on, the drop at b8
already exceeds the predicted band, and the nominal serving batch (b32) cannot
even be resident in the KV pool. The KV-bound long-context regime EXISTS on
MLA at wide-EP attention-DP serving.

## Setup

Exact Phase 61 8-rail recipe (same HEAD, same engine settings, synthetic
non-chat prompts, two-length decode slope, per-DP-rank batch and tok/s); only
deltas: `W7_CTX_TOKENS=16384`, `W7_MAX_MODEL_LEN=20480`, tag `dsv2_8rail_16k`.
Padding verified in the run logs: `prompt_tokens min/mean/max=
16306/16307/16308 (chat=False)`, distinct rotated filler + doc marker per
request. Attention backend: `FLASH_ATTN_MLA` (first-class FA3 MLA decode
kernel, not a triton fallback). Residency verified per point: b8 ran
`Running: 8 reqs, Waiting: 0`, b16 ran `Running: 16 reqs, Waiting: 0`.

## The collapse table (tok/s per DP rank, nospec, greedy)

| batch | 1k (Phase 61 ref) | 16k (this) | 16k/1k drop |
|---:|---:|---:|---:|
| 8  | 283.0  | **180.5 ± 6.9** | **1.57x** |
| 16 | —      | **187.1 ± 1.0** | — |
| 32 | 866.3  | structurally non-resident (see caveat) | — |
| 64 | 1347.7 | (2.5x pool — not runnable) | — |

**Pool arithmetic (H100 80GB, gpu_mem 0.93, ~30 GB/GPU weights).** The engine
pool is **424,624 KV tokens/rank** — at MLA's (512+64) x 2 B x 60 layers =
~70 KB/token that is ~30 GB of latent KV per rank. Demand at ~16.3k
tokens/request: b8 = 130k (31%, resident), b16 = 261k (62%, resident),
**b32 = 522k = 123% of the pool — can NEVER be resident.**

**b32 caveat (structural non-residency).** 16k x b32 at gpu_mem 0.93 is not a
measurable serving point on this sharding: the scheduler runs it as ~17-25
resident + rest queued waves (`Running: 17, Waiting: 15` steady-state, prefix
cache 0%, every method pass re-prefills 531k tokens/rank at ~1.5k tok/s), so
any number produced is preemption-wave queuing, not steady decode. Measuring
16k-b32 would need smaller batch-per-rank or more ranks. The Phase 59
convention for over-pool cells (32k-b64 there) applies: reported as
non-measurable, and **b16 is the clean serving-ish point**.

## Hypothesis test 1 — collapse ratio vs the 1k baselines

- b8: 283.0 -> 180.5 = **1.57x** — above the predicted 1.2-1.5x band (GQA
  showed ~4.3x at b32; the MLA b32 cell is non-resident, see caveat).

## Hypothesis test 2 — batch-scaling at 16k (the decisive one)

- Short context scales strongly on this exact recipe: b8 283 -> b32 866 =
  **3.06x** (and b64 1348, still climbing).
- At 16k: **b8 180.5 -> b16 187.1 = x1.04 (+3.7%) for 2x batch — batch-FLAT.**
  Per-step decode time ~doubles (44.4 ms -> 85.5 ms): the added work scales
  ~linearly with batch, i.e. a per-request-context term dominates the step —
  the KV-bound signature, not compute/comm-bound scaling.

## Mechanism — why wide-EP serving reinforces this regardless of attention architecture

1. **Wide EP shards the FFN, not the KV.** EP16 splits expert weights/compute
   ~1/16 per rank, but attention runs data-parallel: every DP rank reads its
   OWN batch x ctx KV every step. Scaling out shrinks the per-rank FFN slice
   while leaving the per-rank KV read untouched — wide-EP serving pushes the
   long-context step INTO KV-boundness, whatever the attention flavor.
2. **MLA vs modern GQA is only ~1.4x per token.** 70 KB/token latent vs
   Qwen3-30B's 4-KV-head GQA at 96 KB/token. The 10-50x MLA saving is vs MHA;
   vs aggressive GQA it is NOT a regime change.
3. **And the per-context work is not just bytes.** Absorbed-MLA decode does
   128-head x (576-score + 512-value) GEMMs over the whole context (~274
   GFLOP per decoded token at 16.3k) — per-request work that cannot amortize
   across the batch. The bound resource label differs from GQA (attention
   kernel/compute vs DRAM bytes) but the serving phenomenology is identical:
   batch-flat, per-context dominated.
4. **The 236B weights shrink the pool.** ~30 GB/rank of KV headroom vs ~70 GB
   on the 30B GQA testbed — the MLA model hits the capacity wall at a SMALLER
   batch x context product despite smaller per-token KV.

## Verdict

**YES — the KV-bound long-context regime exists on this MLA model at 16k.**
DeepSeek-V2 236B on 2-node EP16 goes batch-flat at 16k (2x batch -> +3.7%),
drops 1.57x at b8 vs the same-recipe 1k baseline, and cannot even hold the
b32 serving batch resident. MLA does not remove the regime that makes
long-context speculative decoding win at serving batch (Phase 59); wide-EP
attention-DP serving reinforces it.

## Bookkeeping

- b8/b16 rows are log-extracted (`data/rows_16k_extracted.txt`;
  `logs/nospec_b8_b16_try1.log`): the invocation was killed at TRY_TO=900s
  mid-b32 and the harness only writes JSON at full completion. Two b32-only
  attempts (1800s, 3600s) were aborted once the non-residency was established
  — no b32 number is reported by design.
- Harness fix committed: `W7_CTX_TOKENS` now pads the default synthetic
  non-chat prompts (previously it required `W7_PROMPT_FILE` and silently
  no-opped). Runner fix: bounded wait on the h106 ssh (orphaned engine
  children hold the pipe open past the remote timeout and wedged the retry
  loop).
- 2 iters + 1 warmup, OUTLEN 160 / SHORTLEN 32; stds: b8 3.8%, b16 0.5%.
- KV pool identical to Phase 61 (424,624 tokens/rank) — max_model_len 20480
  did not change the pool; the 1k references are memory-comparable.
