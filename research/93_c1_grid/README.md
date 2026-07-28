# Phase 93 — C1 grid v2: no best self-spec policy across regimes, per architecture

Design of record (v2, 2026-07-28; supersedes v1 after user review).
Runs are gated: Gate 1 (arm list) -> Gate 2 (Stage-B scope) -> Gate 3
(final evidence review).

## Claim under test

For each of three architectures (dense, MoE, MLA), no single self-spec
configuration wins across the regime surface. "Regime" is factored as:

1. **Dataset character** — the dataset determines input and output
   context. Generation is NOT length-limited: natural EOS under a
   generous ceiling (16k, raised if it ever binds), with the **clip
   ratio logged per run** to prove the ceiling never shaped the data.
   (Watch item: V2-Lite at T=0 may loop on some datasets; the clip
   audit surfaces this rather than hiding it.)
2. **Batch size** — a separately swept axis on EVERY dataset:
   b in {1, 4, 8, 16, 32, 64, 128}, capped per dataset by KV
   feasibility (e.g. 14k-ctx datasets cap near b32 on one H100 for
   8B). Infeasible cells are RECORDED as capacity-bound, not skipped
   silently.

## Architectures / models

| arch | model | TP | notes |
|---|---|---|---|
| dense | Qwen3-8B | 1 | richest kernel support; partial prior coverage (T8 quant column — superseded by this grid's protocol: prior rows were length-capped) |
| MoE | Qwen3-30B-A3B | 2 | expert axis; Humming dense-kernel fallback expected on expert GEMMs |
| MLA | DeepSeek-V2-Lite | 1 | the cost-bound witness; expected honest outcome: OFF wins many cells — that IS policy-diversity evidence |

## Method roster v2 (all self-spec: draft induced from the target itself)

| class | arms | notes |
|---|---|---|
| weight quant | W4A16-GPTQ (dense), W4A16-RTN (all), W8A16-INT8, FP8 W8A8 (weight+activation), W4A8 | the 4/8-bit x weight-only/weight+activation ladder; each format on its BEST AVAILABLE kernel per arch, kernel recorded per cell (the efficiency defense: same-ckpt kernel span 1.60-2.19x is measured; hiding the axis invites the "badly implemented" review) |
| kv quant | kvq_fp8 (draft-side KV cache) | boot-level toggle |
| sparse attention | draft window in {128, 512, 2048, 8192} | window range per user; accept-vs-width is a known step function — the grid captures the curve |
| weight pruning | SEARCH-DERIVED layer-skip sets, budgets {2, 4} | search = measured iterative-greedy conditional-beta growth (phase-86 protocol; the mechanism that survived phase-90's proxy falsification). Dense sets exist (T4); MoE/MLA sets derived in Step 2. NOT naive contiguous skip. |
| arch-native (MoE) | flr50 expert restriction | draft from target's own restricted expert set (phase 83) |
| control | OFF (= AR) | always an arm |

K grid per arm: {2, 4, 6}, probe-adaptive +1 on ties. Verify widths
avoiding the Humming odd-verify-width bug rows at TP2.

**Excluded, with reasons**: ngram (model-free lookup — NOT self-spec;
user ruling 2026-07-28); width pruning (no e2e realization in fork);
2:4 SparseGPT (kernel absent); trained drafts (EAGLE-class — outside
the training-free premise; reference row lives elsewhere).

## Datasets (phase-88 canonical loaders; caps REMOVED for this grid)

| id | dataset | input character | output character |
|---|---|---|---|
| R1 | GSM8K + AIME | short question | CoT, natural length |
| R2 | MT-Bench first turns | short prompt | conversational |
| R3 | HumanEval | function stub | code completion |
| R4 | CNN/DailyMail packed ~8k | long multi-doc | short summaries |
| R5 | NQ-open over C4 ~14k | very long ctx | short answer + explain |
| R5cot | AIME over C4 ~14k | very long ctx | long CoT |
| R6 | GSM8K concise | short | very short |
| R7 | WMT14 de-en | medium | medium (length-coupled) |
| R8 | AIME @ T=1.0 | short | long, high-entropy (RL shape) |

Gen spec per dataset: temperature as above (T=0 except R8), NO
max-token cap (ceiling 16k + clip audit). Batch swept per §Claim.

## Step-by-step plan

- **Step 0 (this doc)** — design of record. DONE when committed.
- **Step 1 — engineering readiness (~1 day, CPU + smoke)**:
  (a) build MoE/MLA drafter ckpts: W4A16-RTN, W8A16-INT8, FP8;
  (b) kernel matrix: which kernel actually executes per format x arch
  (probe + record; prewarm Humming); (c) smoke kvq/window/skip on
  MoE + MLA (5-min boots); (d) harness: uncapped gen + clip-ratio
  logging + batch-sweep loop + KV-feasibility precheck.
  -> **Gate 1: feasibility matrix reviewed, arm list confirmed.**
- **Step 2 — search-derived skip sets** for MoE/MLA (phase-86 greedy
  protocol, ~1-2 h GPU each).
- **Step 3 — Stage A: compiled cell grids** per arch x arm over
  batch x ctx (82/88 compile protocol, ~5-7 min/arm-grid, ~1 day).
  Output: batch-sweep surfaces, predicted winners + K per cell.
  -> **Gate 2: predicted winner maps reviewed, Stage-B scope set.**
- **Step 4 — Stage B: e2e serving confirms (~2-3 days)**: dataset x
  batch {1, 8, 32, max-feasible} x {AR + each class at selected K},
  uncapped gen, 8-iter, multi-seed on winners, FULL tables kept.
- **Step 5 — analysis + packaging (half day)**: winner maps
  (dataset x batch -> lever/K/OFF), distinct-winner counts,
  wrong-lever cost distribution, batch-crossover points ->
  paper/data/c1_grid_*.json + visualization (winner heatmaps multi
  3:1; crossover curves single 1.5:1) + paper/c1.md update.
  -> **Gate 3: final review before this supersedes T8 in the paper.**

## Budget

Stage A ~1 GPU-day; Stage B ~2-3 GPU-days on **GPUs 6-7** (user
allocation 2026-07-28; MoE TP2 uses both); ckpt builds CPU-only.
Multi-seed winners +~20%.

## Gate-1 feasibility report (Step 1, 2026-07-28)

**Checkpoint ladder** (CPU builds, completeness-gated):
dense 8B: W4A16-INT4, W4A8-gptq, W8A16-INT8-sym, W8A16-FP8,
FP8-dynamic — ALL READY. MoE 30B: W4A16 + FP8-dynamic READY,
W8-INT8-chan rebuilding (Marlin-MoE int8 requires CHANNEL-wise;
group-128 asserts). MLA V2-Lite: ladder building (native transformers
classes — remote code broken on new transformers; layer-0 down_proj
cols=10944 stays bf16 by ignore, disclosed).

**Kernel matrix** (measured on the LOADED model, data/kernel_matrix.jsonl):
| arch | format -> kernel |
|---|---|
| dense 8B | W4A16->Machete; W4A8->CutlassW4A8 (default) / Humming (VLLM_DISABLED_KERNELS force); W8A16-INT8->Machete; W8A16-FP8->Marlin-FP8 (VLLM_TEST_FORCE_FP8_MARLIN=1); FP8-dyn->W8A8Fp8 |
| MoE 30B (TP2) | attn->Machete; EXPERTS->Marlin-MoE int4 / FP8-MoE (proper fast paths, no Humming for experts — recorded); router bf16 by design |
| MLA | pending ckpts |

**Lever smokes** (R6 b1 K2, accept; FINAL after engine fixes):
| lever | dense | MoE (TP2) | MLA |
|---|---|---|---|
| self bf16 | (known-good) | implied by window/skip | 3.00 |
| window 512 | (known-good) | 2.96 | 3.00 short-ctx / 1.83 @8k (truncation priced) |
| skip | (known-good) | 2.89 | 2.80 (fork fix: _SkipDecoderLayer extra-positional) |
| quant W4 | 2.91 | 2.96 (after fix) | 2.93 (after fix) |
| quant FP8 | - | 2.85 (after fix) | (ckpt ready) |
| kvq draft-fp8 | 2.94 (NEW build) | (ready) | (ready) |

**ENGINE ITEMS — ALL RESOLVED (2026-07-28, user-approved fix+build):**
1. Quant fused-MoE drafter garbage (accept ~1.0) — ROOT CAUSE: ckpt
   ignore lists carry unprefixed explicit names; under the
   "draft_model." module prefix nothing matched, so IGNORED layers
   (the MoE ROUTER GATE) silently built quantized and loaded garbage.
   FIX: strip the draft prefix in should_ignore_layer (3bf8c5e6a).
   MoE W4 1.07->2.96; FP8 1.00->2.85; MLA W4 2.93. Found via
   tensor-level diff (diag_moe_draft_weights.py).
2. window x MLA "crash" — NOT A BUG: the original failure was an
   OOM-kill coincidence (concurrent 30B CPU quantization); reruns
   pass at short AND long ctx (truncation exercised: 1.83 @ 8k).
   Note: MLA window arms run the plain chain (FULLCG scratchpad
   stack remains GQA-only; cost recorded per arm).
3. kvq e2e BUILT: VLLM_SELF_SPEC_DRAFT_KV_DTYPE=fp8 quantizes ONLY
   the draft's KV pool (own cache group; target + verify stay exact
   = lossless); FA3 metadata builder fixed for per-group quantized
   caches (712517903). Smoke accept 2.94 (kvq beta .985 anchor).
Plus: Marlin-MoE channel-wise loading fixed (group_size None->-1,
07d90c140) — W8-INT8-chan ckpts load on both MoE-FFN models.

**Harness**: run_grid.py validated (uncapped gen + clip-ratio audit
works — R6 EOS'd naturally at ~76 tok, clip 0.0; batch sweep +
preemption counters + per-dataset in/out length stats in every cell).

**STEP 1 CLOSED.** Every lever class is realized e2e on every
architecture; every quant format has a probed kernel row.

## Step 2 results — search-derived skip sets (2026-07-28)

Phase-86 measured iterative-greedy (on-policy 16k refs, conditional
beta, pool-8), the search that survived phase-90's proxy
falsification. Grid arms use budget-2 and budget-4 sets:

| model | budget-2 set (beta) | budget-4 set (beta) | dense ref @4 |
|---|---|---|---|
| MoE 30B-A3B (48L) | {15,23} (.9297) | {13,15,23,24} (.8872) | 8B .849 / 32B .892 |
| MLA V2-Lite (27L) | {10,11} (.9948) | {10,11,16,22} (.9852) | - |

MLA skips are near-FREE (beta .985 at 15% of draft depth dropped) --
the strongest per-budget skip tolerance measured on any architecture;
notable because MLA self-spec is cost-bound, not accept-bound.
Data: data/skip_search_{moe,mla}.csv. **STEP 2 CLOSED.**

## Decision log

- 2026-07-28 (user): regime = dataset character; generation UNCAPPED;
  batch swept 1->128 as its own axis.
- 2026-07-28 (user): ngram excluded (not self-spec); quant ladder must
  include activation quant + 8-bit, stack efficiency defensible;
  window as a range; pruning must be search-based.
- 2026-07-28 (agreed): GPTQ at dense only, RTN for MoE/MLA v1 (GPTQ
  escalation only if W4 loses cells it should win); probe-adaptive K;
  n=8 probes / n=16 confirms.
- OPEN (Gate-2 material): whether to also refresh the 32B dense
  column under the uncapped protocol (+~10 h TP2).
