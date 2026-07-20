# 32B skip-set composition — MEASURED (prediction refuted) — DRAFT

> 2026-07-20, GPUs 4-5 (TP2), serving driver, 3-round medians.
> Skip set {2,4,7,16} = the measured budget-4 iterative-greedy set
> (beta .8924, 86 logs). Lever: VLLM_SELF_SPEC_DRAFT_SKIP_LAYERS
> (index-preserving passthroughs, shared-KV twins intact).

## Results (speedup vs same-GPU AR; accept in parens)

| cell | AR tok/s | W4+win K4* | skip+W4+win K4 | skip+W4+win K5 | Hum K5* |
|---|---|---|---|---|---|
| RKS (KnapSpec b1 prose) | 46.2 | 1.118 (4.46) | **1.143** (4.33) | 1.102 (5.02) | 1.189 (4.84) |
| R1 (b1 math) | 73.4 | 1.26 (4.83) | **1.307** (4.82) | 1.266 (5.77) | 1.45 (5.75) |
| R5cot (b8 14k CoT) | 369.6 | 1.32 (4.62) | 1.377 (4.54) | **1.426** (5.44) | 1.43 (5.27) |

*W4-K4 and Hum-K5 measured on GPUs 0-1 (AR cross-GPU delta ~1%).

## Verdict: the S-model domination prediction was WRONG

- skip x W4 x win strictly improves on W4 x win at ALL measured cells
  (+2.2% RKS, +3.7% R1, +4.3-8% R5cot relative). The prediction
  (composed beta ~.87 -> priced below w4win) assumed multiplicative
  accept composition; MEASURED accept cost of the skip-4 set is
  0.0-0.3 tokens (4.83->4.82 at R1!) -- the greedy set's beta is
  nearly free ON-TASK, the third sub-additivity confirmation.
- KnapSpec h2h footnote upgraded: THEIR lever, composed in OUR
  framework (skip x quant x window), prices the parity cell at
  harness-equivalent ~1.44x -- above their published 1.43x WITHOUT
  the Humming kernel. The kernel lever alone remains stronger
  (~1.51x); skip+Hum composition pending (below).
- Depth interplay: skip prefers K4 at b1 (accept drop not worth the
  deeper chain) but K5 at the deep-decode cell (R5cot 1.426 ~ TIES
  Humming-K5's 1.43 on the cheaper kernel).
- Pool status change: layer-skip moves from "dominated/unbuilt" to
  MEASURED POSITIVE COMPOSITION at 32B -- selected nowhere as the
  headline lever yet (Humming still wins b1 cells), but (a) it ties
  at deep-decode, (b) it remains the zero-extra-memory lever, and
  (c) skip x Hum may stack (arms running; first attempt hit the
  known co-tenant wedge).

## 8B contrast (why the 8B pool verdict stands)

The 8B skip frontier is much weaker (.849 at budget 4 vs .952 int4)
-- the 32B positive composition does NOT transfer down-scale; skip's
value is scale-keyed (deeper stacks tolerate skips), consistent with
KnapSpec's own scale story.
