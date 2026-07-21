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

## skip x Humming: PARKED (co-tenant wedge), quiet-box waiter armed

Both skip+Hum arms wedged at the fresh-compile stage under afternoon
co-tenant load (K5/K4, rc=124; skip machinery itself verified on the
Humming draft: 60-layer shared-KV binding + 240-layer pre-warm ran
clean). Waiter polls for a quiet box (total GPU mem < 2GB) and runs
K5/K4 with 3 tries + worker cleanup (run_skiphum_retry.sh). If the
+3% skip dividend transfers to Humming, the 32B record cells move
~1.45->1.50 (b1 math) and the KnapSpec cell ~1.51->1.55
harness-equivalent.

## skip x Humming: MEASURED (quiet-box waiter, 2026-07-21) — the
## triple composition takes every cell

| cell | plain Hum K5 | skip+Hum K5 | skip+Hum K4 |
|---|---|---|---|
| RKS (KnapSpec b1 prose) | 1.189 | 1.221 (4.90) | **1.245** (4.33) |
| R1 (b1 math) | 1.45 | 1.493 (5.70) | **1.496** (4.81) |
| R5cot (b8 14k CoT) | 1.43 | 1.483 (5.35) | **1.520** (4.52) |

- The skip dividend TRANSFERS to the Humming kernel (+3-6% relative,
  accept cost ~0 again) -- the composition is kernel-independent.
- NEW 32B RECORDS: b1 math 1.50x, deep-CoT 1.52x, and the KnapSpec
  cell at harness-equivalent **~1.57x vs their published 1.43x
  (+10%)** -- their lever, composed with quantization AND windowed
  sparse attention AND the kernel realization, inside our framework.
- The 32B winner class is now a TRIPLE composition: layer-skip x
  weight-quant(W4A8-Humming) x sparse-attention(win512) -- all three
  lever families the user named, measured stacking. The 8B verdict
  (skip priced out at small scale) still stands: composition depth is
  scale-keyed.
- Odd-width note: skip+Hum K4 (width 5) booted clean on the quiet box
  first try -- consistent with contention-multiplied intermittency
  rather than a hard block for K4 (K2 remains 0/6).
