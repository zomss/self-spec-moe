# Baseline survey: the published field, grouped by lever

Record of 2026-08-16. Web survey (arXiv, OpenReview, ACL Anthology, project
repos); **no measurement**. Purpose: enumerate every plausible published
baseline for a training-free, distribution-preserving self-speculative
selector, group each by which of our levers it competes with, rank by threat
and by reproduction cost — and record two findings that force corrections to
this phase's README.

Verification depth varies and is marked per entry. "Verified" means the claim
was read from the paper's own text this session; "unverified" means it comes
from an abstract or secondary summary and must be re-checked before any
number is cited in a paper draft.

---

## Two findings that correct the README

### 1. Sub-block granularity is the family default, not a KnapSpec innovation

Self-SD / Draft&Verify ([2309.08168](https://arxiv.org/abs/2309.08168),
ACL 2024) — the *originating* paper of the layer-skip family — already
searches **attention and MLP sublayers as independent binary variables**
(2L variables for L layers), optimized with Bayesian optimization over
~1,000 iterations (~2.5 h offline for 13B). Verified from the full text;
their Figure 6 plots separate skip distributions for attention (red) and
MLP (blue).

Consequences:

* The README frames independent Attn/MLP selection as what KnapSpec does.
  True, but understated — it is what the **entire family since 2023** does.
  Our whole-layer `VLLM_SELF_SPEC_DRAFT_SKIP_LAYERS` diverges from the
  original 2023 method, not merely from the 2026 one.
* Stage A (sub-block masks) unlocks reproduction of **Self-SD and KnapSpec
  at once** — same engine change, two baselines. It is more load-bearing
  than the README states.

### 2. DEL cannot anchor B3 — it is not training-free

DEL ([2504.05598](https://arxiv.org/abs/2504.05598)) **requires
LayerSkip-trained checkpoints**. Verified from the full text: "We evaluate
DEL on top of LayerSkip using LLaMA-2, LLaMA-3.2, and CodeLLaMA models…
finetuned for LayerSkip by Elhoushi et al. 2024", implementation built on
the LayerSkip codebase. On those checkpoints DEL's own paper reports
**2.16–2.62x over AR** — nowhere near the 0.87x in KnapSpec's table, which
is what early exit produces on a **stock** model whose hidden states were
never trained to be head-readable.

Consequences:

* **B3 as registered-in-draft is unsound.** Reproducing "DEL below 1.0x"
  would confirm only that early exit on an untrained stock model is bad — a
  known fact — not that our reproductions are faithful. See B3' below.
* DEL moves to the **excluded** set (training-required), alongside LayerSkip
  itself.
* The KnapSpec-vs-DEL discrepancy (0.87x vs 2.16–2.62x, a ~2.7x
  disagreement between two published papers on the "same" method) is the
  strongest available evidence for this phase's central risk: baseline
  numbers do not transfer across setups, and a reproduction that silently
  changes the setup produces a number that is wrong in either direction.

---

## The field

### A. Layer-skip drafts — competes with our skip lever

| method | source | granularity | selection rule | own reported speedup |
| --- | --- | --- | --- | --- |
| **Self-SD / Draft&Verify** | [2309.08168](https://arxiv.org/abs/2309.08168), ACL 2024 | sub-block (verified) | offline Bayesian opt, ~1000 iters | up to 1.99x (LLaMA-2 family) |
| **SWIFT** | [2410.06916](https://arxiv.org/abs/2410.06916), ICLR 2025, [code](https://github.com/hemingkx/SWIFT) | layer-level (unverified) | on-the-fly optimization phase, then confidence-aware inference | 1.3–1.6x |
| **CLaSp** | [2505.24196](https://arxiv.org/abs/2505.24196), ACL 2025 | whole layer (unverified) | DP on hidden-state cosine, re-selected after each verification | 1.3–1.7x (Llama3 series) |
| **KnapSpec** | [2602.20217](https://arxiv.org/abs/2602.20217), ICML 2026 | sub-block (verified, prior session) | knapsack with per-block latency weights | up to 1.47x (L3.1-70B, GovReport) |
| **ConfLayers** | [2604.14612](https://arxiv.org/abs/2604.14612), 2026-04, no venue found | sub-block signals (unverified) | adaptive confidence threshold, iterative | not extracted — unverified |

Notes:

* CLaSp is the cosine proxy class **Phase 90 falsified** (rho 0.37 against
  measured acceptance). Reproducing it tests that falsification end to end
  on decode throughput rather than by correlation.
* SWIFT is the closest published shape to our current whole-layer arm; its
  static-set core is nearly what G98-F already measured with knapsack sets.
  The faithful version adds their online re-optimization interval.
* Dynamic-reselection methods (CLaSp, ConfLayers, SWIFT's optimization
  phase) change the skip set at run time — this collides with compiled draft
  paths. See reproduction-cost notes below.

### B. KV-sparsity drafts — competes with our window lever

| method | source | draft KV policy | own reported speedup |
| --- | --- | --- | --- |
| **MagicDec** | [2408.11049](https://arxiv.org/abs/2408.11049), ICLR 2025 | StreamingLLM: sinks + recent window, **fixed budget** independent of context | up to 2x (L2-7B-32K), 1.84x (L3.1-8B), 8xA100, batched |
| **TriForce** | [2404.11912](https://arxiv.org/abs/2404.11912), COLM 2024 | retrieval-selected KV chunks; hierarchical (second, weight-level draft below) | 2.31x (L2-7B-128K, on-chip) |
| **SparseSpec-L** | [2607.27735](https://arxiv.org/abs/2607.27735), 2026-07, AAAI'27 submission | dynamic sparsified + *recallable* KV; recycles verification attention stats; entropy-based speculation-length controller | "up to" — number not in abstract, unverified |

Notes:

* **MagicDec is our window lever.** Sinks + recent window on an unmodified
  self-draft is exactly `VLLM_SELF_SPEC_DRAFT_WINDOW` with 16 sinks. The
  differences are policy-level: they fix the draft KV *budget* constant as
  context grows (ours is a fixed width, which is the same thing), and they
  target the large-batch x long-context regime where KV load dominates.
  Phase 98's activation-threshold measurement (w128: +8.3% slower at b1,
  -3.2% faster at b32) is their bottleneck analysis observed from our side.
  **This baseline is runnable in our engine today** — it is a fixed cell of
  our own lattice, reported under their name and their regime.
* SparseSpec-L (July 2026) is the newest direct competitor; per its related
  work it compares against LayerSkip and MagicDec. Watch it; cite-only for
  now.

### C. Quantized drafts — competes with our w4a16 lever

| method | source | scheme | own reported speedup |
| --- | --- | --- | --- |
| **QSpec** | [2410.11305](https://arxiv.org/abs/2410.11305), EMNLP 2025 | W4A4 draft, W4A16 verify; weights and KV shared across stages | up to 1.80x — **vs a quantized AR deployment**, not FP16 AR |
| **QuantSpec** | [2502.10424](https://arxiv.org/abs/2502.10424), ICML 2025 (Apple/Berkeley) | 4-bit weights + hierarchical INT4/INT8 KV cache draft | up to ~2.5x, >90% acceptance, long context |

Notes:

* QSpec's denominator is a quantized serving stack — its 1.80x is not
  comparable to our "vs stock FP16 AR" convention without conversion.
  Its *verify* path is also quantized (W4A16), so it is not
  distribution-preserving with respect to an FP16 target. Border-excluded;
  cite with the distinction stated.
* QuantSpec is the closest published thing to our quant+window
  *composition* thesis (cheap-weights draft + cheap-KV draft in one
  method). It needs hierarchical quantized-KV kernels we do not have —
  heavy engine work, likely cite-only, but the paper comparison must
  address it precisely because it composes two of our three levers.

### D. Model-free drafters — no draft forward at all

| method | source | mechanism | own reported speedup | availability |
| --- | --- | --- | --- | --- |
| **Prompt lookup / n-gram** | [PLD repo](https://github.com/apoorvumang/prompt-lookup-decoding) | n-gram match against prompt | 2–4x on input-grounded tasks | **in vLLM** (`method="ngram"`) |
| **Suffix decoding** | [2411.04975](https://arxiv.org/abs/2411.04975) (unverified id) | suffix tree over prompt + prior generations, frequency-ranked, adaptive length | not extracted — unverified | **in vLLM** (`suffix`) |
| **Lookahead decoding** | [2402.02057](https://arxiv.org/abs/2402.02057), ICML 2024 (unverified id) | Jacobi iteration + n-gram pool, tree verify | ~1.8x (MT-Bench, paper claim, unverified) | not in vLLM v1 (verify) |

Notes:

* **PLD is a serious threat at exactly R4 and R5.** Summarization and RAG QA
  are input-grounded — the drafted continuation often appears verbatim in
  the prompt. Its claimed 2–4x band sits above anything our selector reaches
  at R4. It costs one config flag to run. If PLD beats the composed selector
  on input-grounded regimes, the honest paper says so and scopes the
  contribution to closed-book/generative regimes — or the selector learns to
  pick PLD as a lever.
* These are drafter *alternatives*, not lever compositions — orthogonal to
  our search space, which is precisely why a reviewer will ask for them.

### E. Excluded, with reasons

| method | source | reason |
| --- | --- | --- |
| LayerSkip | [2404.16710](https://arxiv.org/abs/2404.16710), ACL 2024 | training required (layer dropout + early-exit loss) |
| **DEL** | [2504.05598](https://arxiv.org/abs/2504.05598) | requires LayerSkip checkpoints — see finding 2 |
| Kangaroo | [2404.18911](https://arxiv.org/abs/2404.18911), NeurIPS 2024 | trains an adapter module |
| EESD | — | trains early-exit machinery (unverified; check before citing) |
| Medusa / EAGLE-3 | in vLLM | trained draft heads; **optional trained upper anchor** — one flag if a Qwen3-8B head exists, reported outside the training-free class |
| SpecAttn | [2510.27641](https://arxiv.org/abs/2510.27641), NeurIPS 2025 wksp | **not distribution-preserving** (+15.3% ppl on PG-19) |
| Component-aware hybrid self-spec | [2605.01106](https://arxiv.org/abs/2605.01106) | requires hybrid SSM/attention architecture; Qwen3-8B dense has no such subgraph |
| S3D | [2405.20314](https://arxiv.org/abs/2405.20314) | low-memory-GPU focus, multi-token prediction; check training status before citing |

---

## Threat ranking, updated

1. **MagicDec** — top not because it beats us but because at the lever level
   it *is* us. The reviewer question is "isn't your window lever just
   MagicDec?", and the answer has to be measured, not argued: a MagicDec
   fixed-budget row in every long-context cell, with the selector's value
   shown as composition and regime-switching *on top of* it.
2. **PLD / n-gram** — zero engine work, claims 2–4x on the two regimes where
   our composed number is weakest. The cheapest experiment in the phase and
   the most dangerous to skip.
3. **KnapSpec** — strongest published pure-skip form; the phase namesake;
   needs Stage A.
4. **Self-SD** — same Stage A machinery, and it is the family origin; a
   faithful KnapSpec reproduction that cannot also reproduce Self-SD is
   suspect.
5. **SWIFT / CLaSp** — SWIFT is the cheapest calibration point; CLaSp tests
   Phase 90's falsification end to end.
6. **QuantSpec** — closest to our composition thesis; heavy; may remain
   cite-only but must be engaged precisely.

## Reproduction cost

| tier | baselines | what it takes |
| --- | --- | --- |
| free | ngram, suffix, (EAGLE-3 anchor) | vLLM config flags |
| already ours | MagicDec fixed-budget | a cell of our window lattice, reported under their protocol |
| medium | Self-SD, KnapSpec | Stage A sub-block masks + their offline optimizers (BO / knapsack); static sets, CUDA-graph friendly |
| hard | SWIFT, CLaSp, ConfLayers | run-time skip-set changes collide with compiled draft paths — eager fallback or per-set graph capture; fidelity vs engine-perf tension must be documented per baseline |
| heavy | QuantSpec, TriForce, SparseSpec-L | new KV machinery (quantized hierarchy / retrieval / recallable sparsity) |

## B3': replacing the broken anchor

B3 ("DEL lands below 1.0x") dies with finding 2. Proposed replacement —
**directional anchors** that are robust to the scale and hardware gaps we
cannot close:

* **B3'a** — PLD spikes on input-grounded regimes (R4/R5) and is ~neutral on
  closed-book short regimes (R1/R6), per its own claim.
* **B3'b** — MagicDec-style fixed budget wins at long-context/batched cells
  and loses at short-context batch-1, per their bottleneck analysis (and per
  our own measured activation threshold).
* **B3'c** — each reproduced skip method lands within, or *explainably*
  below, its own paper's reported band — "explainably" meaning the deviation
  is attributed to a measured mechanism (e.g., the 8B draft-loses-8–11%
  floor from `98/results_lever_mechanics.md`), not hand-waved.

All three to be registered with the phase digest barrier before the first
scored boot.

## What this survey does NOT settle

* SWIFT/CLaSp/ConfLayers skip granularity — marked unverified above; read
  the method sections before Stage A design freezes, since if any of them is
  sub-block too, one mask implementation serves five baselines.
* Suffix/Lookahead paper ids and numbers — re-verify before citing.
* Whether an EAGLE-3 head exists for Qwen3-8B.
* How KnapSpec actually ran DEL on Llama3.1-70B (no public 70B LayerSkip
  checkpoint we know of) — worth understanding, not worth blocking on.
