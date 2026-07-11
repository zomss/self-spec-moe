# Phase 77 — the acceptance map: measured β per (lever × architecture × regime)

Source phases: 76 (cost map R + strategy map; E3 proved borrowed β is the weak
link — a 0.03 β error moved a cell prediction ~10%), 18/22/23 (the one-step
rejection-sampling acceptance method this phase generalizes), 17 (layer-skip
collapse, coarse — re-measured here as a curve), 24-25 (local-route β and its
architecture dependence), 74/75 (window/W4 β anchors).

## Objective

Replace every BORROWED β in the strategy map with a MEASURED one:

`β(lever, strength, model, ctx, T)` via **offline teacher-forced rejection-
sampling acceptance** — run the bf16 target over real-text prefixes, run each
lever's draft forward over the SAME prefixes, and per position compute the
spec-decoding acceptance `Σ_x min(p_draft(x), p_target(x))` (sampled, T=1) and
the top-1 match rate (greedy). One model load per (lever, model); no serving
stack, no DP noise, no harness tax — this is what makes ARCHITECTURE BREADTH
affordable, which is the point (MoE-GQA is one architecture among several; the
fabric axis (PCIe) is explicitly out of scope for this phase by user decision).

Then **recompose the strategy map (76 E2) with measured β on both axes** and
re-issue it as v2.

## Why offline one-step β is the right primitive (assumptions, each anchored)

1. **One-step β → multi-token τ via the geometric model** — validated on real
   accepted lengths in P24 B1 (exact to k≤4, 0.97 at k=6-8).
2. **Draft prefix = target KV (SHARED_KV)** — the e2e window draft attends
   sinks+window over the TARGET's KV; the teacher-forced equivalent scores the
   next token with the lever applied to the target-computed prefix. Depth>0
   decay (draft attending its own draft tokens, P28) is NOT modeled — it is
   absorbed by the geometric fit and checked against the E3 anchors.
3. **β has no batch axis** — per-request statistic; offline this is structural.
   The e2e anchors at b8/b32 (P74, 76-E3) carry the batched-numerics side.
4. **Anchor gate (hard)**: the offline β must reproduce the two E3 accept
   lengths through geometric τ within ~3%: dense window K=4 → 4.85/5
   (β≈0.97), MoE window K=6 → 6.49/7 (β≈0.97). If it cannot, the method is
   miscalibrated and NOTHING downstream is valid. (Mirrors how 76-E1 anchored
   TPOT to the P75 slope before sweeping.)

## Grid

**Models** (per-architecture, the phase's main axis; all fit one H100 offline):

| model | architecture | why |
|---|---|---|
| Qwen2.5-7B-Instruct | dense GQA | dense anchor (E3 β measured) |
| Qwen3-30B-A3B | sparse MoE GQA, no shared expert | MoE anchor (E3 β measured) |
| DeepSeek-V2-Lite | MoE MLA, 2 shared experts | MLA + shared-expert axis (P25) |
| GPT-OSS-20B (stretch) | MoE, 32 experts, no shared | small-E routing (P29) |
| Qwen1.5-MoE-A2.7B (stretch) | MoE, high shared mass ~45% | shared-anchor extreme (P25) |

**Levers × strength** (strength is where the new science is):

| lever | strengths | note |
|---|---|---|
| window+sinks | 128, 512, 2048 (sinks 16) | 512 was inherited from P74, never optimized |
| layer-skip | 12.5%, 25%, 37.5%, 50% | the REAL curve; P17 was coarse — either a shallow-skip sweet spot exists (skip25 R≈0.77 needs only β≈0.78 to matter) or the collapse is confirmed as a measured curve for fig5 |
| weight-quant | W4 (dense ckpts exist), fp8 | β vs bits; P22 anchor 0.92/0.95 |
| KV-quant fp8 (draft-only, EMULATED) | e4m3 | offline we CAN quantize only the draft's read of the prefix KV — measures the β of the hypothetical draft-only KV pool whose R-reference column the map already carries |
| local-route top-C | 0.25E, 0.5E, full (MoE models only) | P24/25 masking; the lever whose β is known to swing across architectures (shared-expert mass) — the portability claim |

**Regime axes**: ctx {2k, 16k, 32k} prefixes from real text (on-dist prompts +
long documents for 16k/32k — asset to secure: P57 prompts are short; need a
long-doc source, e.g. concatenated c4/Omni-MATH already in the HF cache);
temperature {greedy top-1, T=1.0 overlap} — both fall out of the same scored
distributions at no extra forward cost.

## Pre-registered predictions (falsification targets)

1. **β is lever-intrinsic and ctx-stable for window under shared KV** (P74:
   4.76@16k vs 4.79@32k) — but MAY degrade with ctx for window WITHOUT the
   sinks (control arm).
2. **Layer-skip β collapses super-linearly with skip fraction** (P17) — the
   curve should pass below β≈0.78 (skip25's break-even from the 76 map) before
   25% skip. If it does not, a new map region opens.
3. **Quant β is monotone in bits and ctx-flat** (P22/23 one-step numbers).
4. **Local-route β is the ONLY lever whose β is architecture-unstable**
   (swings with shared-expert mass, P25: +0.13 to +0.58 anchor lift); window
   and quant β are portable within ±0.02 across models. This is the map's
   "portable vs non-portable acceptance" claim.
5. **Temperature costs little** for high-β levers (P75: 3.77→3.69 accept) and
   proportionally more for low-β ones.

## Method / commands (to be built)

- `scripts/score_accept.py` — loads (model, lever, strength); iterates prefixes
  at each ctx; teacher-forced forward; writes per-position acceptance +
  top-1 match to `data/beta_<model>_<lever>_<strength>_c<ctx>.json`.
  Levers implemented as forward-time deltas on ONE loaded model where possible
  (window = attention mask; skip = layer slice; local-route = router mask from
  P24/25 code; quant = second load or fake-quant hooks per P22/23).
- `scripts/anchor_gate.py` — geometric τ from measured β vs the two E3 accept
  lengths (HARD GATE, run first on the two anchor cells).
- `scripts/recompose_map.py` — 76's E2 with `beta.csv` replacing the BETA table;
  emits strategy map v2 + a diff vs v1 (which cells changed winner).
- Box: cloud-9ezI3Q, GPUs 0/1/6/7 only, caches on /data (see memory + 76
  env_e76.sh; offline scoring needs 1 GPU per model ≤30B).

## Decision criteria

- **Gate**: anchor gate passes (±3% on both E3 cells) before any sweep cell.
- Phase SUCCEEDS if the recomposed map v2 (a) keeps its validated cells within
  the E3 error band, and (b) resolves the two open β questions: the skip curve
  (point cloud on fig5) and the local-route portability claim (prediction 4).
- Any prediction refuted → law refinement, documented (the E4/76 pattern).

## Expected next artifact

`results_accept.md` + `data/beta.csv` + strategy map v2 (+ fig5 updated with
the measured skip curve). Phase 78 then fits the term-decomposed cost model to
76's `summary.csv` with 77's β as the fixed accept side.
