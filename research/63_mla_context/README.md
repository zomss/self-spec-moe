# Phase 63 — Does MLA remove the long-context KV-bound regime that GQA models show?

## Source

- **Phase 59** (`research/59_multicontext/results_multicontext.md`) measured, on
  the real 2-node EP16 fabric, that Qwen3-30B-A3B (GQA, 96 KiB/token KV)
  **collapses 1581 -> 368 tok/s at b32 going 2k -> 16k** and goes batch-flat:
  at long context the decode step is KV-read-bound at any batch (the MagicDec
  regime), which is exactly where long-context speculative decoding wins.
- **Hypothesis (this phase):** DeepSeek-V2 236B (MLA, ~67.5 KiB/token latent KV
  — tiny relative to its ~4x-larger per-token compute) does NOT collapse: at
  16k it stays compute/comm-bound, throughput still scales with batch, and the
  drop from short context is only ~1.2-1.5x. This decides whether the
  long-context KV-bound spec-decode regime exists on MLA-class models at all.

## Setup

- DeepSeek-V2 236B (`deepseek-ai/DeepSeek-V2`), 2-node TP2 x DP4/node = EP16
  (h107+h106), the exact Phase 61 8-RAIL recipe (`NCCL_IB_HCA='^mlx5_8'`,
  gpu_mem 0.93, CG sizes 8,16,32,64,128, max_num_batched 2048, synthetic
  non-chat prompts, greedy, two-length decode slope OUTLEN 160 / SHORTLEN 32,
  2 iters + 1 warmup). Phase 61 same-HEAD 1k references: **283.0 / 866.3 /
  1347.7 tok/s at b8/32/64**.
- ONLY deltas vs Phase 61: `W7_CTX_TOKENS=16384` (pads every synthetic prompt
  to ~16.3k tokens with rotated filler + per-request doc marker -> distinct
  per-request KV), `W7_MAX_MODEL_LEN=20480`, tag `dsv2_8rail_16k`, port 14100+.
- Harness fix (this phase, `research/52_two_node_e2e/scripts/w7_2node.py`):
  `W7_CTX_TOKENS` previously required `W7_PROMPT_FILE` and silently no-opped on
  the synthetic default; now it synthesizes the bank from the same synthetic
  prompts. Padding pre-verified offline: 16306-16308 tokens, 32/32 distinct.
- Mode nospec only, batches 8,16,32 (`scripts/run_16k_236b.sh`).

## Memory check (H100 80GB, gpu_mem 0.93, weights ~30 GB/GPU)

MLA KV = (kv_lora 512 + rope 64) x 2 B x 60 layers = **67.5 KiB/token**.
KV pool at this recipe: **~424.6k tokens/rank** (~27.3 GiB; Phase 61 engine
log). Per-rank KV demand at ~16.5k tokens/request:

| batch | KV tokens | vs pool |
|---:|---:|---|
| 8  | ~132k | 31% — clean |
| 16 | ~264k | 62% — clean (the serving-ish point) |
| 32 | ~529k | **125% — EXCEEDS the pool** -> vLLM queues/preempts; reported with the Phase 59 near-pool caveat |

So b16 is the clean serving-ish point; b32 carries a pool caveat (same
convention as Phase 59's 16k-b64 / 32k-b32 cells).

## Readout

See `results_mla_context.md`: the 16k-vs-1k collapse table, the two hypothesis
tests (collapse ratio at b8/b32; batch-scaling at 16k), and the verdict on
whether the KV-bound long-context regime exists on MLA.
