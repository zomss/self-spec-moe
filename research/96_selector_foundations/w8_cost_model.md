# W8 — constructive Round-1 cost model (pre-registration)

User direction (2026-08-06): (1) fold the verify-step expert-coverage
excess into Round 1, with offline profiling of kernel and serving-stack
efficiency per lever; (2) measure the cheap-target condition in Round 1
— when the target step is cheap and no draft can undercut it, OFF is
the standard verdict, decided before lever enumeration.

## 1. The generalized identity

C2's S = (1+fK)/(KR+1) assumes verify costs one target step. True for
dense (weights are read once regardless of token count); FALSE for
sparse activation below expert-coverage saturation, where K+1 verify
tokens activate ~min(E, (K+1)k) experts. Generalized:

    S = (1 + fK) / (K·D/T + V(K,b) + C/T)

- T(b)   target decode step time (AR)
- D(b)   draft forward step time (lever-dependent)
- V(K,b) verify step cost in target-step units (=1 dense;
         coverage-scaled for MoE)
- C(b)   per-armed-step serving-stack overhead (lever-independent)

The W7/C2 inversion R̂ = (τ/S−1)/K books V's excess and C into R —
right verdicts, conflated mechanism (and the source of the oracle's
P-W7d miss and the b32 regime spread in R̂).

Expected-coverage law (i.i.d. routing): cov(n) = E·(1−(1−k/E)^n) for
n routed tokens/step. Qwen3-30B-A3B: E=128, k=8, 48 layers. V2-Lite:
E=64, k=6 (+2 shared always-on, first layer dense), 27 layers.
Saturation at b·(K+1) ≳ 32 tokens — predicting the observed b32
crossing. Real routing is correlated; the law is validated, not
assumed (P-W8c is the test).

## 2. Round-1 funnel (amended design)

- **Stage 0 — cheap-target cell test**: from T(b) (off-arm), measured
  C, coverage V, and a bytes-roofline draft floor D_min (active
  quantized bytes / peak BW): S_max = (1+K)/(K·D_min/T + V + C/T).
  S_max ≤ 1 proves OFF for the whole cell — no draft needs to exist.
  Can only prove OFF, never ARM (fail-closed preserved).
- **Stage 1 — (lever, K)-pair elimination**: constructive
  break-even with profiled per-lever D. K-aware: coverage depends on
  b·(K+1), so different K sit at different saturation points.
- **Round 2 unchanged**: measures f on survivors; sole authority to ARM.

## 3. Measurement protocol (W8a)

Built-in profiler `vllm.v1.spec_decode.self_spec_profiler`
(VLLM_SELF_SPEC_PROFILE=1, coarse mode: draft_chain / draft_forward /
verify clean of fine-timer perturbation; cpu_* regions for C;
incremental flush survives force-kill; read the per-PID file with the
most samples = rank-0).

Boots: arch ∈ {mla, moe} × b ∈ {1, 8, 32} × K ∈ {1, 4}, R2 only,
uncond, notune, ITERS=2, single batch per boot (region samples must
not pool across batches). MLA boots pair-parallel on GPU0/GPU1; MoE
TP2 sequential. Profiled boots are NOT scored serving artifacts (the
region syncs perturb e2e rate); only region times are used.

T(b) comes from W7 off-arm rates (T = b/rate). Disclosed bias: e2e
includes prefill, so T is overestimated → D/T, C/T, V underestimated
→ biases TOWARD spec viability. Stage-0 OFF proofs are a fortiori
sound; ARM claims are never made from this pass.

## 4. Pre-registered predictions

- **P-W8a (verify excess)**: V(4,b1) ≥ 2 for both arches; V decreases
  monotonically in b; V(4,b32) ≤ 1.3.
- **P-W8b (kernel/dispatch term)**: at b1, D/T exceeds the
  bytes-roofline ratio by ≥ 2× (the draft step is dispatch-bound, not
  bandwidth-bound); at b32 it is within 1.3× of roofline.
- **P-W8c (reconstruction, the headline)**: the assembled denominator
  K·D/T + V + C/T reproduces W7's measured τ/S at K=4 within ±10%
  for the R2 cells (b1, b8, b32) of both arches. This validates the
  coverage law and licenses the constructive Round 1.
- **P-W8d (Stage-0 works)**: the floor construction proves OFF at
  b1/b8 for both arches with no lever enumeration. Disclosed
  expectation: the stack floor C ALONE is insufficient (C/T at b1 is
  ~0.1-0.2, not ~1); the proof needs the bytes floor + coverage terms.
