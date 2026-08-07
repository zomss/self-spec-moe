# W13 — does τ* reproduce the dense selector? (pre-registration)

Written BEFORE any W13 run. The redesign (W12 §2.0) says Round 1 should
emit a required-acceptance label

    τ*(ℓ,K,c) = K·D(ℓ,c)/T(c) + V(K,c) + C(c)/T(c)

computed from cost alone, after which `S = τ_measured / τ*`. W8
validated that form out-of-sample on MLA/MoE at b32 (+1.5% / −0.3%).
This tests it on **dense — the architecture that ships** — against a
selector decision we have already measured (W6).

## Why this is the right next experiment

Everything downstream depends on Round 1 being constructive rather than
tabulated: the continuous cell index, the closed-form window crossover,
the span-τ* pool criterion, and the τ*-ordered ladder fallback. All of
it rests on τ* being computable offline and correct. It is currently
verified on two sparse architectures at one batch. If it fails on
dense, the redesign does not ship.

## Design

Model Qwen3-8B, draft `~/ckpts/Qwen3-8B-W4A8-gptq`, K=4, notune,
ITERS=4, per-regime fixed length (constant batch width, W9 rule).

Regimes span the (batch, ctx) space in the axes that matter:

| regime | batch | prompt | fixed gen | cell character |
|---|---|---|---|---|
| R1 | 1 | ~83 | 1024 | short ctx, b1 |
| R5 | 8 | ~14037 | 512 | long ctx, b8, short gen |
| R5cot | 8 | ~14086 | 3072 | long ctx, b8, LONG gen |

Batches 1 and 8 are both swept for every regime, giving 6 cells.

Pool (5 compositions, spanning the window axis so the flatness claim is
testable — `w-off` is full-KV):

    s-none/w512   s-none/w2048   s-none/w-off   s-2,8/w512   s-2,8/w2048

Three boot classes:
1. **off** ×1 → `T(c)`
2. **profiled uncond** ×5 → `D, V, C` (region times; W8 §3b showed these
   predict the UNPROFILED cost, so they are sound inputs)
3. **unprofiled uncond** ×5 → measured `τ` and the actual `S`

11 boots, TP1, two GPU lanes. The profiled and unprofiled boots are
separate on purpose: the profiled boot's own serving numbers are
discarded (pre-registered rule since W8), so the actual `S` must come
from an unperturbed run.

## Pre-registered predictions

- **P-W13a (the point test)**: predicted `S = τ/τ*` matches the
  measured `S` within **±10%** on ≥5 of the 6 cells.
- **P-W13b (the decision test — the one that matters)**: the
  **ranking** of the 5 compositions is reproduced exactly in each
  regime, i.e. the selector built from τ* picks the same winner Round 2
  measured. A model that gets every level wrong but every ORDER right
  still ships; the converse does not.
- **P-W13c (cost is content-free)**: R5 and R5cot share batch and
  nearly share input context but differ 6× in generation length. τ* for
  the same composition should agree between them to within the ctx
  difference alone (≈2.5k of 14k). Large disagreement would mean cost
  carries content, refuting the dichotomy's cost half.
- **P-W13d (windowed D is flat in ctx)**: `D(w512)` changes < 10%
  between R1 (~1k ctx) and R5 (~14.5k ctx), while `D(w-off)` grows
  markedly. This is the premise behind the closed-form window
  crossover; if D(w512) grows with ctx the crossover is not solvable
  and the fine cell index must be tabulated after all.
- **P-W13e (span beats top-N)**: the 5 compositions' τ* values at a
  given cell spread by ≥15%, i.e. the pool genuinely offers distinct
  operating points rather than clustering. If they cluster, the
  "span τ*" pool criterion has nothing to span and the ladder's
  fallback chain is cosmetic.

## Disclosed

W6 measured these same compositions at these regimes under natural EOS
in the S_dec currency; W13 uses fixed length in the step currency, so
the two are not directly comparable and W13 is scored against its OWN
unprofiled arms. Consistency with W6's ranking is reported as a
secondary observation, not a test.
