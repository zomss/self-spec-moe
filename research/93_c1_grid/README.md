# Phase 93 — C1 grid: method x regime x architecture (COLLECTION; no runs yet)

Goal (user, 2026-07-28): show that for the three architectures (dense,
MoE, MLA) there is NO best self-spec policy across regimes. Step 1 =
collect the representative methods and the per-regime datasets; runs
are GATED on explicit go.

## 1. Architectures and models

| arch | model | why this one | draft ckpts on disk |
|---|---|---|---|
| dense | Qwen3-8B | richest kernel support (Humming/Cutlass/Marlin/Machete); T8 column partially exists | W4A16-INT4, W4A8-gptq, W8A16-FP8, sparse24 |
| MoE | Qwen3-30B-A3B (A3B active) | measured in T2/T3 (window 1.15x, flr50 flip); EP/expert axis | NONE (must build) |
| MLA | DeepSeek-V2-Lite | measured in T2/T3 (self-spec loses at accept 5.9 -- the cost-bound witness) | NONE (must build) |

## 2. Representative method per lever class (the roster)

One representative per class, chosen for (a) class-representativeness,
(b) runnability in the fork on ALL three arches, (c) prior
measurement anchors:

| class | representative | realization in fork | dense | MoE | MLA | anchor (T3 beta) |
|---|---|---|---|---|---|---|
| weight quant | W4A16-INT4 drafter (GPTQ if ckpt exists, else RTN-sym; llmcompressor) | separate quantized ckpt as draft_model; kernel auto (Machete/Marlin) | YES (ckpt exists; also W4A8-Humming as kernel-flip arm) | build ckpt (~30-60 min CPU) | build ckpt | .952 / .972 / .997 |
| kv quant | kvq_fp8 (draft-side KV cache fp8) | boot-level draft kv_cache_dtype | YES | YES | YES | .985 / .971 / .997 |
| sparse attention | win512 (draft KV window) | fork window lever (hot-switchable) | YES | YES | YES | .975 / .971 / .922 |
| weight pruning | greedy layer-skip set, budgets {2,4} (per-model frontier, T4) | VLLM_SELF_SPEC_DRAFT_SKIP_LAYERS (index-preserving) | YES (budget sets known) | YES (sets to derive: ~30 min beta screen) | YES (sets to derive) | frontier known for dense only |
| (arch-native pruning variant) | flr50 expert restriction | MoE-only (phase 83) | - | YES (measured 1.03x flip / 0.63x wrong-cell) | - | .953 |
| (model-free control) | ngram | MLA-relevant (best measured spec there, 0.78x) | optional | optional | YES | - |

Excluded and why (recorded so reviewers see the pool was principled):
- width pruning (ffn25 etc.): beta measured (T5b) but e2e realization
  never built (died pre-e2e by build-on-selection); not runnable.
- 2:4 SparseGPT: kernel absent in the fork (marlin_24 combo pending).
- EAGLE/Medusa-class trained drafts: out of scope by the paper's
  training-free premise (kept as a reference row elsewhere).

Per-lever K grid: K in {2, 4, 6} (T8: shallow-K is load-bearing;
deeper saturates). OFF/AR is always an arm. Verify widths avoid the
Humming odd-width bug rows on TP2.

## 3. Regime -> dataset matrix (canonical, phase 88 loaders)

All loaders exist in research/88_regime_eval/scripts/regime_datasets.py
(HF datasets cached under /data). Prompt formatting adapts per model
tokenizer (chat template where the model has one; V2-Lite = base-style).

| regime | dataset (real, prior-work-grounded) | shape | gen spec |
|---|---|---|---|
| R1 math CoT | GSM8K test + AIME 1983-2024 | b1 | 1024 tok, T=0 |
| R2 conversation | MT-Bench first turns | b1 | 512 tok, T=0 |
| R3 code | HumanEval | b8 | 512 tok, T=0 |
| R4 summarization | CNN/DailyMail 3.0.0, packed to ~8k ctx | b8/8k | 512 tok, T=0 |
| R5 RAG QA | NQ-open questions over real C4 doc context ~14k | b8/14k | 512 tok, T=0 |
| R5cot RAG CoT | AIME problems over C4 context ~14k | b8/14k | 3072 tok, T=0 |
| R6 burst | GSM8K concise answers | b32/2k | 256 tok, T=0 |
| R7 translation | WMT14 de-en test | b8 | 512 tok, T=0 |
| R8 RL rollout | AIME, T=1.0 (EfficientRollout shape) | b16 | 2048 tok, T=1.0 |

(RKS KnapSpec-parity cell exists as a 10th loader; used for h2h, not
part of the no-best-policy grid.)

## 4. What already exists vs what is NEW

- Dense column: T8 gives 9-regime winners for the QUANT lever family
  (W4A8-Hum x K) + skip compositions at 32B. Missing at 8B per-regime:
  win512, kvq_fp8, skip-set as standalone arms. So even dense needs
  the 3 remaining classes run per-regime.
- MoE column: NOTHING per-regime (only 16k/2k cells). Needs full grid
  + W4 ckpt build + skip-set beta screen (frontier not measured).
- MLA column: NOTHING per-regime. Same needs. Expectation from cells:
  many regimes -> OFF/ngram wins (that IS the finding: the best
  "policy" on MLA is mostly not to draft -- diversity includes OFF).

## 5. Proposed run protocol (for go/no-go — NOT started)

Per arch: for each regime x lever-class arm, short probe at K
{2,4,6} (2 iters) -> pick K -> 8-iter confirm for AR + every class at
its best K. Report per-regime winner + full class table (not just
winners) since the claim is about the whole policy surface.

Rough budget (serving runs ~8-12 min each; probes ~3 min):
- dense 8B (3 new classes x 9 regimes): probes ~2.7 h + confirms ~5 h => ~8 h, GPUs 0/1
- MoE 30B (4 classes + flr50): ckpt build ~1 h CPU + skip screen ~0.5 h + ~12 h (TP2 => GPUs 0+1 both)
- MLA V2-Lite (4 classes + ngram): ckpt build ~0.5 h + ~9 h (TP1)
Total ~30 GPU-hours wall over ~3-4 days interleaved. Multi-seed on
winners only (+~20%).

Open decisions for the go call:
1. Dense at 8B only, or also refresh the 32B column with the two
   missing classes (kvq, win512 per-regime)? (+~10 h TP2)
2. GPTQ or RTN for the MoE/MLA W4 ckpts? (RTN = fast + matches
   EfficientRollout Tier-0; GPTQ = +0.13 beta at 32B but hours of
   calibration; proposal: RTN now, GPTQ only if W4 loses cells it
   should win)
3. K grid {2,4,6} everywhere, or add K3/K5 where probes are close?
   (proposal: probe-adaptive, +1 K only on ties)
4. n=8 prompts per regime (phase-88 default) or n=16 for tighter
   per-regime numbers? (proposal: n=8 probes, n=16 confirms)
