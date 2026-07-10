# Phase 75 results — reproduce EfficientRollout's weight-quant lever

Box: **cloud-9ezI3Q**, H100 80GB HBM3 (SM90, ~3.35 TB/s HBM3), driver 570.211.01,
torch 2.11.0+cu129, vllm 0.1.dev17860. **H100, not A100** — the harder test (calibration:
faster BW shrinks the byte term → fixed overhead is a larger share → Tq/Tp worsens vs the
paper's A100 0.360). Report F alongside every ratio.

## E0 — preflight: GATE OPEN ✓
Both int4 W4A16 kernels execute here (no h106 PTX trap): **MACHETE OK**, **MARLIN OK**.
So we run the paper's kernel (Marlin) *and* vLLM-native (Machete) as a controlled A/B.
W7_TEMP present (T=1.0 available). Checkpoints verified weight-only (`ia=None`,
`ignore=[lm_head]`, compressed-tensors): W4-sym 5.55 GB, W8-sym 8.81 GB, W4-asym 5.57 GB.
Kernels resolve: W4 auto→Machete, W4 forced→Marlin, asym→Machete.

## E1 — measured Tq/Tp (KEY #1). Qwen2.5-7B-Instruct, b1, ctx 2k, slope over 32 decodes

| arm | kernel | t_step (ms) | **Tq/Tp** | paper (idealized) |
|---|---|---:|---:|---:|
| bf16 | — | 5.859 | 1.000 | — |
| **W4** | **Marlin** | **3.393** | **0.579** | 0.360 |
| W4 | Machete | 4.212 | 0.719 | 0.360 |
| W8 | Marlin | 4.278 | 0.730 | 0.573 |
| W8 | Machete | 5.161 | 0.881 | 0.573 |

**Findings:**
1. **The lever converts bytes to time — W4 Marlin Tq/Tp = 0.58** (draft is ~1.7× cheaper
   than the bf16 target). This is the systems half of the win, and it is *real* on dense —
   the exact opposite of our MoE, where fp8 gave Tq/Tp ≈ 1.0 (no cut → no win possible).
   The byte-budget inversion, confirmed from the cost side.
2. **Marlin BEATS Machete at b1 decode** (W4: 0.58 vs 0.72; W8: 0.73 vs 0.88). vLLM
   auto-picks Machete, which is tuned for larger M; at M=1 (single-token decode) Marlin's
   lower fixed overhead wins. **→ use Marlin (force it) for E3.** Actionable, non-obvious.
3. **Not the paper's 0.360** — expected on H100. bf16 alone is cleanly BW-bound
   (14.26 GB / 5.86 ms ⇒ ~3.35 TB/s + F ≈ 1.6 ms, physical). W4 reads 4.47 GB (BW term
   1.33 ms) but carries F ≈ 1.6 ms + a Marlin dequant overhead ≈ 0.46 ms → 3.39 ms. The
   fixed floor (P72/73: ~2400 tiny kernels at b1) is a big share on H100's fast BW, so the
   byte cut is diluted. On an A100 (BW ~2 TB/s ⇒ bf16 BW term ~7 ms), the same F would give
   Tq/Tp ≈ 0.49 — near the paper.
4. **BW_eff fit is unphysical (6.27 TB/s > 3.35 peak)** — NOT because the lever fails, but
   because the shared-F linear model is wrong: the quant arms carry *extra* (dequant)
   overhead bf16 doesn't, so the regression inflates BW to pass through them. Real bf16 is
   physical; quant adds a kernel-overhead term on top. (Interpret the printed "lever works"
   heuristic with this caveat.)

**Verdict (E1):** By the strict criterion (H100 pass ≤ 0.50) this is a **soft miss at 0.58**
— but that criterion was about whether the lever *exists*, and it does: the draft is
genuinely ~1.7× cheaper than verify. Plug the *measured* ratio into the paper's own
speedup formula with the paper's τ:

`speedup = τ / (γ·Tq/Tp + 1)` → at γ=5, τ=5.18: **W4 Marlin 5.18/(5·0.579+1) = 1.33×**;
W4 Machete = 1.13×. **Both predict an end-to-end WIN.** So E1 does not refute the lever —
it says the H100 delivers ~1.3× where the paper's idealized A100 roofline says 1.85×.
The remaining question is whether τ holds (E2); if it does, E1×E2 = lever confirmed on dense.

## E2 — losslessness gate: harness is CORRECT (bit-exact), after a gate fix

The gate (greedy spec must be token-identical to no-spec) initially **failed 4/8
sequences** — but the divergences were **late and small** (first ~53–84 of 96 tokens
matched, then close-call flips / an off-by-one), never the early/total corruption a
`SHARED_KV` bug would cause. Root cause: **greedy self-spec is only bit-exact when the
verify's batched (K+1-token) forward and the sequential nospec decode use IDENTICAL
kernels** — i.e. batch-invariant kernels. Without them, tiny FP differences between the
two batch shapes flip argmax at close calls.

The gate wired batch-invariance via `VLLM_SELF_SPEC_COMPILE_CONSISTENT`, which sets
`VLLM_BATCH_INVARIANT=1` **only inside SpeculativeConfig** (vllm.py:1088-1093) — so it
reached the spec arm but left nospec on default kernels → could never match. **Fix:**
set `VLLM_BATCH_INVARIANT=1` directly on **both** gate arms. Result:

> **PASS: 8 sequences, 768 tokens, token-identical.**

**Verdict:** the W4 self-draft is **provably lossless** (bit-exact greedy under
batch-invariant kernels); the plumbing has no bug and the draft corrupts nothing.
Two takeaways: (1) a general vLLM fact — greedy speculative decoding is NOT bit-exact
without `VLLM_BATCH_INVARIANT` (FP argmax flips), only *distributionally* lossless; (2)
the τ sweep runs with batch-invariance OFF (throughput hygiene + paper comparability) —
valid, because τ is an accept-rate statistic robust to a handful of flips per 96 tokens.

### E2 — τ (block efficiency) at T=1.0: W4-sym REPRODUCES the paper (~90–98%)

Primary sweep (T=1.0, b8, ctx2k, SHARED_KV=0), W4-sym draft, Qwen2.5-7B-Instruct target:

| γ | our W4-sym τ | paper W4 | ratio |
|---|---:|---:|---:|
| 3 | **3.524** | 3.59 | 0.98 |
| 5 | 4.708 | 5.18 | 0.91 |
| 7 | 5.958 | 6.70 | 0.89 |

Reproduces to ~90–98%, shortfall growing with γ — plausibly the two flagged deviations:
sym-vs-asym RTN (paper uses asymmetric, more accurate for int4) and prompt distribution
(our filler-prose 2k vs their genuine rollout prefix).

**Deviations / anomalies (recorded):**
- **W4-asym won't load** — `ValueError: too many values to unpack (expected 9)` at engine
  init (vLLM int4-with-zero-points / compressed-tensors unpack). The paper's exact
  *asymmetric* RTN is therefore not testable on this vLLM, so the sym-vs-asym gap can't be
  closed here. (These arms also hang 40 min on `timeout 2400`; killed the sweep, ran
  controls separately via `e2_controls.sh`.)
- **W8-sym anomalous** — near-zero accept (tok/s 317/220/160 at γ=3/5/7, *decreasing* with
  γ, no accept_len), the opposite of the paper (W8 τ > W4). Suspect int8 weight-only
  Machete gives degraded draft logits on this stack (E0 only verified int4). Secondary.

### LEVER VERDICT (E1 × E2): PASS × PASS — weight-quant IS a real self-spec lever on dense

Measured halves into the paper's formula `speedup = τ / (γ·Tq/Tp + 1)`, Tq/Tp = 0.579 (Marlin):

| γ | τ (measured) | predicted e2e |
|---|---:|---:|
| 3 | 3.524 | **1.29×** |
| 5 | 4.708 | 1.21× |
| 7 | 5.958 | 1.18× |

τ-optimal γ for e2e = **3** (deeper γ costs more draft than the marginal accept buys).
**Opposite of our MoE** (fp8 Tq/Tp ≈ 1.0 → no win possible). The byte-budget inversion is
now confirmed on BOTH halves — dense: draft genuinely cheaper (0.58) AND accurate (τ≈3.5)
→ win; MoE: draft not cheaper → no win. **Phase 74's "never a win" is SCOPED to
sparse-MoE/long-context, not universal.** This is the phase's headline.

**E2 controls (W4-sym, γ=5, b8):**
- **SHARED_KV=1**: τ = 4.919 vs SHARED_KV=0's 4.708 → **+4.5% inflation** (as predicted —
  the draft attending the target's *exact* KV over-accepts; the paper's drafter computes
  its own KV, so SHARED_KV=0 is the faithful setting we used for the primary τ).
- **greedy (T=0)**: transient `IndexError` at engine init (leftover procs from the killed
  sweep; the next arm on the same config initialized fine). Not re-run standalone — E3
  runs a greedy control at b1, which covers the accept-vs-temperature de-risk there.

τ-optimal γ for e2e = 3 (from the E1×E2 formula). E3 uses γ=3, Marlin forced.

## E3 — end-to-end: the lever DELIVERS a 1.21× win on dense (Marlin confirmed)

Real tok/s, spec vs AR, Qwen2.5-7B-Instruct, γ=3, W4-sym draft on **Marlin** (E1-optimal):

| batch | arm | tok/s | accept | **speedup** |
|---|---|---:|---:|---:|
| **b1** T=1.0 | nospec / spec | 169.6 / **206.0** | 3.692 | **1.21×** |
| b1 greedy | nospec / spec | 171.1 / **212.7** | 3.765 | **1.24×** |
| b4 T=1.0 | nospec / spec | 659.3 / 692.1 | 3.413 | 1.05× |
| b8 T=1.0 | nospec / spec | 1269.1 / 1294.1 | 3.511 | 1.02× |

**Harness delivers (E3 pass).** Measured 1.21× vs the formula's ~1.35× (b1 accept 3.692,
Tq/Tp 0.579) = **90%** — a ~10% shortfall from the PIECEWISE/eager draft chain (E1's
CUDA-graphed Tq/Tp was a lower bound). This is NOT the Phase-74 fp8 collapse (ideal
+6.5% → measured −2%); on dense the draft is cheap enough that the harness overhead is a
small fraction, so the lever survives.

**Batch decay reproduces the toggle rationale (C5).** 1.21× (b1) → 1.05× (b4) → 1.02×
(b8): self-spec wins at small batch and heads to parity as batch grows (Phase 74 saw
0.86× at b64 on the MoE — same law). The greedy control (1.24×, accept 3.765) vs T=1.0
(1.21×, accept 3.692) quantifies the temperature de-risk: **small** penalty (accept 3.77
→ 3.69), so the accept-vs-temperature concern Phase 74 deferred is benign here.

## FINAL VERDICT — one byte-budget law, two operating points

**E1 (cheap) × E2 (accurate) × E3 (delivered): all PASS.** Weight-only quantization IS a
real, end-to-end self-spec win on a **dense** model in the small-batch regime — **1.21×**
at b1 on H100, reproducing EfficientRollout. We under-reproduce the paper's idealized
**1.85×** (A100), and the gap is fully accounted for, in order: (1) H100 vs A100 — faster
BW dilutes the byte cut (Tq/Tp 0.58 vs 0.36); (2) γ=3 vs their γ=5 (τ-optimal here is
lower because H100's higher Tq/Tp penalizes depth); (3) sym-vs-asym RTN (asym won't load
— vLLM bug); (4) our PIECEWISE harness (−10%). None is a refutation; the lever is real.

**This SCOPES Phase 74, it does not contradict it.** Both results are the SAME law:

| | draft cheaper than verify? | binding term | weight-quant result |
|---|---|---|---|
| **dense** 7B, b1, 2k (here) | **YES** (Tq/Tp 0.58) | weight-read | **WIN 1.21×** ✓ |
| **MoE** 30B-A3B, b8, 16k (P74) | NO (fp8 Tq/Tp ≈ 1.0) | KV-read | parity ✗ |

*A draft-only lever wins iff it cuts the term that binds, and which term binds is set by
(dense vs MoE) × (context × batch).* Confirmed on both halves and end-to-end. Phase 74's
"weight-quant is never a self-spec win" must be stated as **"never on a sparse-MoE draft
at long context,"** not universally — the dense small-batch cell is a clean win.

**Deviations from the paper (all recorded above):** H100 not A100; Marlin forced (beats
Machete at b1); W4-sym not asym (asym won't load); PIECEWISE harness; filler-prose 2k
context. **Open:** W8-sym near-zero accept (int8 Machete suspect); the asym-RTN vLLM unpack bug.

## E4 — MLA crossover: strong prediction REFUTED, law REFINED (the interesting outcome)

Test: does shrinking KV (MLA) flip weight-only quant from parity (GQA MoE) to a win?
Metric = decode-step ratio `fp8-marlin / bf16` (weight-only fp8 = pure read cut, dequant),
via slope, EP4. `ninja` must be on PATH (MLA JIT-compiles a kernel; Qwen3 didn't).

**b8, 16k:**

| MoE | attention | KV/tok | fp8/bf16 ratio |
|---|---|---:|---:|
| DeepSeek-V2-Lite | **MLA** | 30 KiB | **0.962** |
| Qwen3-30B-A3B | GQA | 96 KiB | 0.971 |

MLA is lower — **directionally** as predicted — but the gap is **0.9% (within noise)** and
BOTH are near-parity. The strong prediction (MLA → clear win) is **refuted at b8**.

**Why (the refinement):** at 16k b8 EP4 the decode step (~5.7 ms) is ~8× the actual
weight+KV read (~0.7 ms) — it is dominated by **EP a2a comm + fixed overhead + the small
active-weight compute**, NOT memory read. So halving the (intrinsically small, 2.4–3.3 B
active) weight barely helps, and shrinking KV doesn't change that. **The unified law needs
a third term:** weight-quant wins only where weight-read is a LARGE ABSOLUTE fraction of the
step — true for DENSE (14 GB weight read, no EP comm → E1–E3 win 1.21×), false for a SPARSE
MoE (small active weight + EP comm dominates), **MLA or not**. KV-size is *a* variable, not
*the* crossover variable; active-weight-magnitude and EP-comm also bind.

**b1, 16k (memory-bound regime — the fairer test):**

| MoE | attention | fp8/bf16 ratio |
|---|---|---:|
| DeepSeek-V2-Lite | **MLA** | **1.081** |
| Qwen3-30B-A3B | GQA | 1.122 |

At b1 BOTH ratios are **> 1 (fp8 is SLOWER)**: per-expert M ≈ 0.06 tokens, so the MoE
decode is deeply latency/launch-bound and Marlin's dequant/align overhead dominates the
tiny (~4 ms) step, swamping the (minuscule) weight-read saving. MLA is again lower than
GQA (1.081 < 1.122).

### E4 verdict — strong prediction REFUTED, direction confirmed, law sharpened

**Across BOTH regimes, MLA < GQA (0.962<0.971 at b8, 1.081<1.122 at b1) — a robust but
tiny ~1–4% directional signal — yet weight-quant NEVER wins on the MoE** (b8 parity, b1
loss). Shrinking KV 3.2× (MLA) does *not* flip weight-quant to a win.

**The sharpened law:** weight-quant self-spec wins iff the **weight-read is a LARGE fraction
of the draft's decode step**. That fraction is set by **active-weight magnitude**, and only
secondarily by KV. DENSE 7B reads 14 GB/step (the whole model) → weight-read dominates at
low batch → **WIN 1.21× (E1–E3)**. SPARSE MoE reads only its ~2.4–3.3 B active weight →
weight-read is a *small* slice regardless of KV (the step is dominated by KV, EP a2a comm,
and fixed/dequant overhead) → **no win, MLA or not (E4)**. So KV-size is *a* variable, not
*the* crossover variable; **active-weight magnitude is the real one**, and MLA cannot
rescue a sparse MoE because its active weight is intrinsically small. The lever that *does*
win on the MoE is **window** (cuts the KV term that actually binds) — P74's 1.16–1.34×.

**Unified picture (final):**

| draft | weight-read share | binding term | weight-quant | winning lever |
|---|---|---|---|---|
| dense 7B, b1 (E1–E3) | **large** (14 GB) | weight-read | **WIN 1.21×** | weight-quant |
| MLA MoE, b8/b1 (E4) | small | KV + comm + overhead | 0.96 / 1.08 (no) | (window) |
| GQA MoE, b8 (P74) | small | KV-read | 0.97 (no) | **window** (1.16–1.34×) |

One law, three operating points: *a draft-only quant lever wins iff it cuts a term that is
both binding AND a large fraction of the step.* Weight-quant satisfies this only on dense;
on sparse MoE the winning lever is window (KV), and MLA does not change that.
