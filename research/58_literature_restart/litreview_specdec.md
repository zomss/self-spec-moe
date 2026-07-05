# Speculative Decoding for LLM Inference — Literature Review (2023–2026)

**Scope.** Rigorous, current review of speculative decoding (SD) for a systems project on
**spec decoding in communication-bound multi-node MoE serving**. Emphasis on: how cheap the
*draft* is, how expensive the *verify* is, and the **regime** (does it help at high/serving batch
= throughput, or only at low batch = latency?), plus distributed / expert-parallel awareness.

All arXiv IDs below were verified against real arXiv abstract/HTML pages or search returns; none
are invented. Venue tags are given where a paper is published. A handful of 2026-dated IDs
(`26xx.*`) are recent preprints (current date: mid-2026) and are flagged as such.

**Framing used throughout.** Modern GPU decode at low batch is **memory-bandwidth-bound**:
arithmetic intensity is ~1–2 FLOP/byte, two-to-three orders of magnitude below the hardware
"ridge point" (~400–600 FLOP/byte on H100/H200/B200). All that idle compute is the *speculation
budget* — a target forward over `γ+1` positions costs ≈ one normal decode step, so accepted draft
tokens are nearly free. As batch (or, for long context, KV-cache size) grows, decode moves toward
**compute-bound**, the budget shrinks, and verifying extra draft tokens starts to *tax* throughput.
This memory-vs-compute crossover is the axis on which every "does SD help at serving batch"
question turns.

---

## 1. Foundations: vanilla speculative decoding / speculative sampling

The two 2023 papers established the **draft-then-verify with lossless rejection sampling** paradigm.
For each drafted token `x` with draft prob `q(x)` and target prob `p(x)`: accept with prob
`min(1, p(x)/q(x))`; on rejection, resample from the normalized residual `max(0, p−q)`. The
marginal of every emitted token is exactly `p`, so the **output distribution is identical** to
standard autoregressive sampling from the target — the speedup is "free" (no quality loss, no
target fine-tuning). This losslessness guarantee is the bedrock the whole field preserves (or
explicitly relaxes).

| Method (year, venue / arXiv) | Core mechanism | DRAFT cost | VERIFY cost | Speedup | Regime |
|---|---|---|---|---|---|
| **Blockwise Parallel Decoding** — Stern, Shazeer, Uszkoreit (NeurIPS 2018; **1811.03115**) | `k` auxiliary heads predict positions `t+1…t+k`; one scoring pass keeps the longest matching prefix. Precursor to SD. | `k` cheap linear heads on shared trunk (near-free) | 1 scoring forward over `k` tokens (**linear chain**) | ~2× fewer iters (greedy) | Latency; single model |
| **Speculative Decoding** — Leviathan, Kalman, Matias (ICML 2023; **2211.17192**) | Small separate draft model proposes `γ` tokens autoregressively; target scores all `γ` in one forward; modified rejection sampling accepts a prefix. | `γ` sequential passes of a *small separate* model | 1 target forward over `γ+1` tokens (**linear chain**) | 2–3× | **Low-batch/latency only**; extra draft FLOPs erode gains as batch grows; not multi-node aware |
| **Speculative Sampling** — Chen et al., DeepMind (2023; **2302.01318**) | Independent development on Chinchilla-70B; formalizes the modified rejection-sampling scheme that preserves the target distribution. | `γ` passes of small draft | 1 target forward over drafted continuation (**chain**) | 2–2.5× (Chinchilla-70B) | Low-batch/latency; no arch/param changes |
| *Survey* — Xia et al. "Unlocking Efficiency… A Comprehensive Survey of Speculative Decoding" (ACL Findings 2024; **2401.07851**) | Taxonomy of drafters + verification strategies; third-party benchmark (Spec-Bench). | — | — | — | Reference anchor |

**Takeaways.** (1) Losslessness is a *rejection-sampling* property, not an approximation — it is what
distinguishes SD from lossy parallel decoding. (2) The **separate-draft-model** approach is
model-agnostic but requires maintaining a second model and paying its FLOPs; this cost is what the
self-draft line (§2) removes. (3) Both foundational papers verify a single **linear chain**;
**trees** (§3) came later.

---

## 2. Self-draft / trained-head methods

These grow drafting *into* the target via cheap heads/features, avoiding a separate model and
usually reaching higher accept lengths. Key distinction: **independent** heads (each future
position predicted without seeing earlier draft tokens: Medusa, Meta-MTP) vs **sequentially
dependent** (position `i` conditions on `<i`: Hydra, EAGLE, ReDrafter, DeepSeek-MTP) — the latter
achieve higher accept lengths.

| Method (year, venue / arXiv) | Core mechanism | DRAFT cost | VERIFY cost | Accept len / speedup | Regime |
|---|---|---|---|---|---|
| **Medusa** — Cai et al. (ICML 2024; **2401.10774**) | Multiple extra decoding heads on frozen backbone's last hidden state; candidates = Cartesian product of heads, verified via **tree attention**. | Several parallel linear heads (near-free); *independent* | 1 backbone forward over a **token tree** (~64 cands) | Medusa-1 >2.2×; Medusa-2 2.3–3.6× | Latency/low-batch |
| **Hydra** — Ankner et al. (2024; **2402.05109**) | Fixes Medusa's independence: **sequentially-dependent** heads (each conditions on prior draft tokens). | Sequential heads (still cheap) | 1 forward over candidate **tree** | 2.70× vs AR; 1.31× vs Medusa | Latency/low-batch |
| **EAGLE** — Li et al. (ICML 2024; **2401.15077**) | Autoregress at the **feature (2nd-to-top layer) level** + one-step-ahead token to resolve feature uncertainty; reuse target LM head. Lossless. | 1 lightweight AR feature head | 1 target forward over drafted **tree** | ~3× (MT-bench) | Latency/low-batch |
| **EAGLE-2** — Li et al. (EMNLP 2024; **2406.16858**) | Adds **context-aware dynamic draft trees**: draft confidence (a good accept-prob proxy) expands/prunes per context. | EAGLE head + dynamic tree | 1 forward over ~48–60-token dynamic tree | 3.05–4.26× | Latency/low-batch |
| **EAGLE-3** — Li et al. (NeurIPS 2025; **2503.01840**) | **Drops feature-prediction loss** (capped data scaling); direct token prediction + **multi-layer feature fusion** ("training-time test"). | EAGLE-style head on fused low/mid/high features | 1 forward over dynamic tree | up to 6.5×; **1.38× throughput @ batch 64 (SGLang)** | Latency **+ explicitly claims serving-batch gains** |
| **MTP (Multi-Token Prediction)** — Gloeckle et al., Meta (2024; **2404.19737**) | Pretrain with `n` **independent** heads predicting next `n`; primarily a training objective, heads enable self-speculative drafting. | `n` parallel independent heads (near-free) | 1 forward over `n` proposed tokens | up to ~3× self-spec | Training-time method; inference benefit low-batch |
| **DeepSeek-V3 MTP module** — DeepSeek-AI (2024; **2412.19437**) | **Sequential MTP modules keeping the full causal chain** at each depth; depth-1 module used in production serving. | Sequential MTP module | 1 forward; drafted 2nd token verified | **85–90% accept of 2nd token → ~1.8× TPS**; lossless | **Production serving** (vLLM/SGLang); designed for deployment |
| **ReDrafter (Recurrent Drafter)** — Zhang et al., Apple (2024; **2403.09919**) | Single lightweight **RNN draft head** on hidden state; **dynamic tree attention over beam search** dedupes prefixes; KD from target. | 1 small RNN over beam candidates | 1 target forward over dynamic tree | up to 3.5× (H100); 2.3× (Apple MLX) | Latency/on-device + server; TensorRT-LLM support |
| **Falcon** — Gao et al. (AAAI 2025; **2412.12639**) | **Semi-autoregressive** self-drafter: `k` tokens/forward via LSTM + relaxed causal mask + MLP; Coupled Sequential Glancing Distillation; custom tree. 2 transformer layers. | 1 SAR pass emits `k` tokens (fewer draft passes) | 1 target forward over custom tree | multi-× (stronger intra-block deps) | Latency/low-batch |

**Takeaways.** Accept length is driven mostly by sequential dependence and by drafting on *features*
(EAGLE) rather than tokens. **DeepSeek-V3's MTP** is the production-relevant anchor for your project:
a single sequentially-dependent head, ~85–90% second-token acceptance, ~1.8× TPS, already integrated
into serving stacks. **EAGLE-3** is the rare self-draft method to explicitly report a
serving-batch (batch-64) throughput gain. None of these are *inherently* distributed/multi-node
algorithms — distribution is handled by the surrounding framework, not the SD method.

---

## 3. Tree / structured drafting and token-tree verification

**Origin.** Token-tree verification originates with **SpecInfer** (2305.09781): draft candidates
are organized as a **token tree** sharing prefixes, verified in a **single** target forward using a
**topology-aware causal mask** ("tree attention") — each token attends only to its ancestors, with
DFS KV traversal. Medusa independently popularized the same trick. Essentially all later methods
(EAGLE-2/3, Sequoia, OPT-Tree, …) inherit this mask.

| Method (year, venue / arXiv) | Core mechanism | VERIFY cost — how it scales with tree size | Speedup / accept | Regime |
|---|---|---|---|---|
| **SpecInfer** — Miao et al. (ASPLOS 2024; **2305.09781**) | Boosted SSMs → merged token tree; topology-aware mask; DFS KV reuse. | Whole tree (dozens of tokens) in 1 forward; **no explicit width-scaling analysis** | 1.5–2.8× distributed; ~4 tok/step | Latency; **distributed-aware** (TP+PP; "only tokens communicated → negligible comm") |
| **Sequoia** — Chen et al. (NeurIPS 2024; **2402.12374**) | DP-optimal tree topology + **hardware-aware optimizer** picks size `n`, depth `d`. | **The canonical scaling result** (below). Trees up to 512+ nodes. | 4.04× (7B), 2.27× (33B); up to 9.96× offloaded 70B | Latency, on-chip **and** offloading; hardware-aware |
| **EAGLE-2** — (EMNLP 2024; **2406.16858**) | Dynamic tree via expand (top-k by global accept prob) + rerank. | ~48–60 tokens, depth 6, ancestor-only mask | 3.05–4.26×; τ≈4–5.5 | Latency |
| **OPT-Tree** — (TACL 2025; **2406.17276**) | Adaptive tree maximizing **E[accept length]** per step. | Node count can exceed **500** → up to **10 tok/step** (70B) | up to 3.2× | Latency |
| **DySpec** — (2024; **2410.11744**) | Greedy runtime tree expansion using draft-prob↔accept-rate correlation. | Dynamic tree | 9.1× throughput / 9.4× latency (low-T, 70B) | Latency + throughput |
| **ProPD** — (ICCAD 2024; **2402.13485**) | **Early pruning** of unpromising branches + dynamic tree generation. | **Explicitly balances verify compute/parallelism across batch sizes** | 1.1–3.2× | **Batch-size-aware** (small→large) |
| **Recursive Speculative Decoding (RSD)** — (2024; **2402.14160**) | Sampling **without replacement** (Gumbel-top-k / stochastic beam) to maximize tree diversity. | Diverse tree at fixed compute budget | beats baselines at fixed draft len | Latency |
| **SMART** — (2026 preprint; **2604.09731**) | Marginal **benefit–cost** rule: expand a node only if its ratio > current tree-level speedup. | **Directly models super-linear verify cost**; stops at hardware saturation | +15.4% (LLM) / +20% (MLLM) extra | **Compute-bound batching**, multi-GPU-aware |
| **Traversal Verification** — (2025; **2505.12398**) | **Leaf-to-root** sequence-level tree verification (vs top-down token-level); lossless. | Same tree, recovers subseqs top-down discards | higher accept len | Verify-algorithm; any regime |
| **Batch Speculative Decoding Done Right** — (2025; **2510.22876**) | EQSPEC/EXSPEC fix **ragged-tensor** desync (position IDs/mask/KV) across a batch. | Realignment overhead **super-linear in batch** (≤40% of compute @ bs 8) | up to 3× (bs 8 vs 1), ~95% output-equiv | **Batched correctness** |

**How VERIFY scales with tree size (the load-bearing result for wide-vs-narrow trees).**
Early work implicitly assumed a target forward over a token tree costs **O(1)** regardless of tree
size. **Sequoia empirically refutes this**: forward time `t(n)` grows **non-linearly (≈linearly)**
with tree tokens `n`. Its speedup model is explicit:

> **Speedup(n,d) = G(n,d) / [ t(n) + d·c ]**, and for batch `b` the denominator becomes **t(b·n) + d·c**,

where `G` = expected accepted tokens, `c` = per-step draft cost. Sequoia proves the numerator grows
only **logarithmically** — `G(n) ∈ Ω(log n / log log n)` — while `t(n)` grows ~linearly, so there is
a **finite optimal tree size `n\* < ∞`**; picking it right yields up to 38% over fixed-size configs.
This is the canonical wide-vs-narrow analysis. The tradeoff **flips with regime**:
- **Memory-bound (batch=1, latency):** a wide tree is nearly free (amortizes weight streaming), so
  wide trees win (OPT-Tree >500 nodes, EAGLE-2 ~48–60 tokens).
- **Compute-bound (large batch):** verifying a big tree *per sequence* pushes past peak FLOPs, verify
  cost grows **super-linearly**, and speedup can go **negative**. SMART formalizes this "efficiency
  paradox"; ProPD attacks it via early pruning; Batch-SD-Done-Right exposes the ragged-tensor
  super-linearity. **MagicDec (§4) is the long-context exception** where trees still help at batch 32–256.

**Comm/cost of trees, distributed.** SpecInfer is the most explicit: the request manager and GPU
workers "**only communicate tokens, not vector representations**," so tree verification adds
*negligible* communication. The batched hazard is the **ragged-tensor problem** (Batch-SD-Done-Right)
whose realignment overhead scales super-linearly with batch.

---

## 4. The batching / throughput question (central) — does SD help at serving batch?

This is the crux for a throughput-bound MoE serving project. The literature has converged on a
**two-dimensional crossover: arithmetic intensity, driven by batch AND context length**, not batch
alone.

| Paper (year, venue / arXiv) | Core mechanism | Batching finding: help at serving batch? Crossover / why |
|---|---|---|
| **The Synergy of Speculative Decoding and Batching** — Su, Giannoula, Pekhimenko (2023; **2310.18813**) | Quantitative model: optimal speculation length vs batch size; adaptive length. | **Canonical "it fades at large batch."** Optimal spec length is a *decreasing function of batch* (verify shifts memory→compute-bound); at large batch it collapses toward no/low speculation. |
| **MagicDec** — Sadhukhan, Chen et al. (ICLR 2025; **2408.11049**) | Fixed-context-window (StreamingLLM-style) draft; can even self-speculate the target since KV, not weights, is the bottleneck. | **Counters conventional wisdom: SD DOES help at large batch for long context.** KV-cache loading scales with `batch × seqlen`, doesn't amortize like weights → decode stays memory-bound at *any* batch for long context. Up to 2.51× at batch **32–256**, 32k–100k ctx, 8×A100. Speedup *increases* with batch. |
| **SmartSpec / TurboSpec (goodput)** — Liu et al. (2024; **2406.14066**) | "Goodput" (verified tok/s) metric; dynamically sets per-request spec length, down to **0**. | **Explicit disabling at high load.** SD stops helping past request rate ~12 (5 tokens), *degrades* past ~16 (3 tokens); larger models cross over sooner. Up to 3.2× latency cut at low/moderate load. |
| **SpecDec++** — Huang et al. (2024; **2405.19715**) | Trained acceptance-prediction head; stop speculating when P(≥1 reject) exceeds threshold (MDP). | Adaptive length; 2.04–2.26× (Llama2 7B/70B). "Don't over-speculate" lever. |
| **DISCO (Dynamic Speculation Lookahead)** — Mamou et al. (2024; **2405.04304**) | Classifier decides per step: keep drafting or verify. | Adaptive lookahead ~10% over best static; motivates that fixed lookahead is suboptimal. |
| **BASS (Batched Attention-optimized Spec Sampling)** — Qian et al. (Findings ACL 2024; **2404.15778**) | Custom CUDA kernels for ragged draft attention; parallel across batch AND draft-token dims. | **Makes SD work at batch>1 at all.** Batch 8 (7.8B, A100): peak decode util 15.8% (~3× regular decode). Modest-batch regime; kernel work is non-trivial. |
| **TriForce** — Sun, Chen et al. (COLM 2024; **2404.11912**) | Hierarchical: sparse-KV self-draft + smaller model; targets long-context KV bottleneck. | Long-context/offloading latency (128K on 2×4090, 7.8×). Same KV-bound logic MagicDec generalizes to batch. |
| **Ouroboros** — Zhao, Huang et al. (EMNLP 2024; **2402.13720**) | Training-free phrase-level drafting from a candidate pool. | Argues generation is memory- not compute-bound (the enabling premise); up to 2.8×. |
| **Sarathi-Serve** — Agrawal et al. (OSDI 2024; **2403.02310**) | Chunked-prefill + stall-free batching to fill decode-time compute budget with prefill work. | *Not SD*, but the key adjacency: it spends the **same idle-compute budget SD targets** on piggybacked prefill. Under continuous batching, SD spec length **competes with prefill tokens** for a fixed per-iteration token budget. |
| **MoESD: Unveil SD's Potential for Sparse MoE** — Huang et al. (2025; **2505.19645**) | Theoretical model of SD tradeoffs specific to MoE memory movement. | **MoE-specific, positive at moderate batch.** MoE benefits *more* than dense at medium batch; **sparser MoE → broader batch range where SD helps** (expert FFNs are memory-bound with idle compute, so verifying draft tokens is nearly free). |
| **Utility-Driven SD for MoE ("Cascade")** — Saxena et al. (2025; **2506.20675**) | "Speculation utility" = token-gain / verify-cost; selectively enables SD, tunes K. | **MoE-specific caution.** Draft tokens collectively activate **more experts** → 2–3× more weight movement / verify time; naive SD can **slow MoE by up to 1.5×**. Gates SD on/off. |

> *Illustrative (industry blog, not peer-reviewed):* "Decode Is Memory-Bound. Speculation Is the
> Arbitrage" gives ridge points (H100 FP8 ≈591, H200 ≈412, B200 ≈562 FLOP/byte), single-stream
> decode at 1–2 FLOP/byte, short-context break-even ≈ **batch 32**, and a context threshold (~1k
> tokens on B200) beyond which decode stays memory-bound at any batch. Consistent with the papers;
> treat numbers as illustrative.

**Consensus (see §7a).** (1) Short context: SD helps at small batch, is net-negative past
crossover ≈ **batch 32** (compute-bound); serving systems should **disable it under high load**
(Synergy, SmartSpec). (2) The crossover is really about *arithmetic intensity*, so **long context +
large batch keeps decode KV-bound and SD helps at serving scale** (MagicDec). (3) For **MoE** the
answer is genuinely mixed and must be measured — SD can help *more* than dense at moderate batch
(MoESD) or *hurt* via extra expert activation (Cascade). The design the literature converges on is a
**utility/goodput-gated, batch-adaptive speculation length**.

---

## 5. Verify-cost reduction — is there any *lossless* way to make VERIFY cheaper?

The verify-cost literature splits into three tiers by what it touches and whether it stays lossless.

| Method (year / arXiv) | What it reduces | Lossless? | Regime / MoE-aware |
|---|---|---|---|
| **Staged Speculative Decoding** — Spector & Ré (2023; **2308.04623**) | Draft (draft-the-draft, restructured spec batch) | **Lossless** | Latency, small-batch/on-device; dense |
| **Cascade Speculative Drafting** — Chen et al. (NeurIPS 2024; **2312.11462**) | Draft (vertical + horizontal cascades of drafters) | **Lossless** | Latency; dense |
| **Draft & Verify** — Zhang et al. (ACL 2024; **2309.08168**) | Draft (self-spec via **layer skipping**) | **Lossless** | Latency; dense; ~1.99× |
| **LayerSkip** — Elhoushi et al. (ACL 2024; **2404.16710**) | Draft (early-exit self-spec; **reuses KV/activations**) | **Lossless** | Latency; dense; up to 2.16× |
| **Kangaroo** — Liu et al. (NeurIPS 2024; **2404.18911**) | Draft (shallow sub-net + adapter; double early-exit) | **Lossless** | Latency; 88.7% fewer params than Medusa |
| **SWIFT** — Xia et al. (2024; **2410.06916**) | Draft (on-the-fly adaptive layer-skip; training-free) | **Lossless** | Latency; dense |
| **Lookahead Decoding** — Fu et al. (ICML 2024; **2402.02057**) | **Removes the draft model** (Jacobi n-gram + parallel verify) | **Lossless** | Latency; scales with #GPUs |
| **REST** — He et al. (NAACL 2024; **2311.08252**) | Draft (retrieval datastore supplies drafts; tree verify) | **Lossless** | Latency; 1.62–2.36× |
| **Prompt Lookup / n-gram** — Saxena (open-source; in vLLM/HF) | Draft (copy n-grams from prompt) | **Lossless** | Input-grounded tasks; 2–4× |
| **SpecDec++** — (2024; **2405.19715**) | **Verify** (fewer tokens verified via adaptive length) | **Lossless** | Latency |
| **HiSpec** — (2025/26; **2510.01336**) | **Verify** (cheap early-exit *intermediate verifier* rejects bad drafts before full target; periodic re-validation) | **Lossless** | Throughput; dense |
| **FASER** — (2026 preprint; **2604.20503**) | **Verify** (per-request spec len + **early pruning of rejected tokens inside verification**) | **Lossless** | Throughput/serving; dense |
| **EVICT — "Making Every Verified Token Count"** — Pan et al. (2026 preprint; **2605.00342**) | **Verify** (**truncate draft tree before target verify**, keep cost-effective prefix ⇒ fewer unique experts loaded) | **Lossless** | Throughput; **MoE-aware**, SGLang; up to 2.35× |
| **SP-MoE** — Chen et al. (2025; **2510.10302**) | **Communication / offload** (SD-aware expert prefetch + pipelined offload masks CPU–GPU transfer) | **Lossless** (hides, not eliminates) | Throughput; MoE offloading |
| **SpecMoE** — Bang et al. (2026 preprint; **2604.10152**) | Draft+verify / bandwidth (self-assisted SD for MoE) | **Lossless** | Throughput up to 4.3×; MoE memory-constrained |
| **SpecMoEOff** — Wang et al. (2025; **2508.21706**) | **Communication / offload** (enlarge expert workload to hide offload latency; CPU verify kernel) | **Lossless** | Throughput; MoE offloading |
| **MoE-SpeQ** — Wang et al. (2025; **2511.14102**) | **Communication / I/O** (draft predicts *which experts* future tokens need → proactive prefetch) | **Lossless** (spec guides prefetch only) | Latency; MoE edge/offloaded |
| **Faster Cascades via Speculative Decoding** — (2024; **2405.19261**) | Whole draft+verify (cascade deferral) | **Lossy** (quality/cost tradeoff) | Latency/quality |
| **SpecPV** — Tan et al. (2025; **2512.02337**) | **Verify** per-token cost (verify with **partial KV states**) | **Lossy** ("minor degradation") | Long-context latency |
| **MoE-Spec** — McDanel et al. (2026 preprint; **2602.16052**) | **Verify** memory/comm per token (**verify-time expert budgeting**: fixed expert capacity/layer, drop long tail) | **Lossy** (drops rarely-used experts) | Throughput; MoE-aware; vs EAGLE-3 |
| **Sparse Computation in Verification** — (2025 preprint; **2512.21911**) | **Verify** FLOPs (sparsify attention + FFN + **MoE experts** at verify) | **Lossy** (approximate) | Long-context + MoE |

**Critical question — can the VERIFY forward be made cheaper *losslessly*?**

**No known method losslessly lowers the *per-token* verify forward of a *committed/accepted* token
below one full target forward pass.** That component is treated as essentially **irreducible** under
exact (distribution-preserving) SD. What *is* reducible losslessly:

1. **The NUMBER of verified tokens/branches** — adaptive length (SpecDec++), utility-gating (Cascade,
   2506.20675), dynamic-tree truncation (EVICT). Draft-free variants (Lookahead, REST, prompt-lookup)
   and self-spec (Draft&Verify, LayerSkip, Kangaroo, SWIFT) cheapen the **draft**, but still run the
   **full model** as verifier. Token-tree verification batches candidates into one forward but does
   **not** lower the FLOPs of any accepted token.
2. **WASTED verify work + MoE redundant expert loading/communication**, without touching accepted-token
   compute. HiSpec rejects doomed drafts via a cheap intermediate verifier while staying lossless by
   periodically re-validating accepted tokens against the target; FASER prunes rejected tokens
   mid-verification. **In MoE, verify cost is dominated by loading the *union of experts* across the
   draft tree** — EVICT cuts that losslessly by tree-branch pruning; SP-MoE / SpecMoEOff / MoE-SpeQ
   losslessly *hide* (prefetch/overlap) the CPU–GPU / offload transfer. **This is the most active
   lossless frontier and the one most relevant to a MoE-aware angle.**
3. **Per-token verify compute/comm directly** — the moment you cap experts (MoE-Spec), use partial KV
   (SpecPV), or sparsify the verify forward (2512.21911), you become **lossy**.

**Bottom line:** you cannot cheapen an accepted MoE-verify token's own forward *losslessly*; you can
(i) verify fewer tokens, (ii) reject bad ones before the full pass, and (iii) in MoE, eliminate
redundant expert loads / hide the transfer. **A lossless method that lowers the *intrinsic*
compute-or-communication of a committed MoE verify token (e.g., verifying with a strict subset of
experts and provably identical output) does not appear to exist yet — genuinely open territory.**

*Adjacent (NOT spec decoding):* **"Speculative MoE" → retitled "Semantic Parallelism"** (Li et al.;
**2503.04398**) losslessly cuts EP **all-to-all** volume via speculative token–expert *co-scheduling*
(offline co-activation clustering + online rebatching/reshuffling). It is a communication-efficient
MoE *scheduling* method, not a draft/verify SD method, but it is the closest existing work that
treats the token↔expert mapping as an all-to-all-communication object — a useful prior-art anchor.

---

## 6. Distributed / multi-GPU / EP-aware speculative decoding

| Paper (year, venue / arXiv) | Distributed relation (TP/EP/PP/disagg) | Comm-aware? | Regime |
|---|---|---|---|
| **SwiftSpec** — Async disaggregated SD + fused kernels (ASPLOS 2026; **2506.11309**) | **Disagg + TP.** Names "**communication overheads under small-batch tensor-parallelism**" as why naive SD+TP fails; disaggregates draft/target onto separate GPU groups. | **Yes — strongest comm/TP-aware example.** | Latency; 8×H800/H200 |
| **EasySpec** — Layer-parallel SD (2025; **2502.02493**) | **TP.** Idle GPUs help during draft (draft's optimal TP < base's); KV recalibrated post-verify. | Partially (no extra verify comm) | Latency; multi-GPU single node |
| **Efficient SD for Llama at Scale** — Meta (2025; **2508.08192**) | **TP.** Production EAGLE (tree attn, multi-round) on 8×H100; 1.4–2.0× **at large batch**. | Kernel-aware, not comm-modeling | Latency + throughput; single node |
| **PipeInfer** — Async Pipelined Speculation (SC24; **2407.11798**) | **PP / cluster.** Continuous async speculation + early cancellation; explicitly targets **low-bandwidth interconnects**. | **Yes** (cancellation cuts wasted bandwidth) | Latency, single-request; **multi-node** |
| **FlowSpec** — Continuous Pipelined SD (2025; **2507.02620**) | **PP.** Score-based step-wise verify + dynamic draft expansion over stages; "reduces comm overhead in multi-node inference." | **Yes** | Latency; **multi-node** pipeline |
| **SpecPipe** — Pipeline-Parallel SD (2025; **2504.04104**) | **PP.** Fills pipeline with speculative tokens (branch-prediction analogy); ideally 1 tok/pipeline step. | Bubble/p2p-aware | Latency; multi-GPU/-node PP |
| **DSI — Distributed Speculative Inference** — Timor et al. (ICLR 2025; **2405.14105**) | Multi-instance **"speculation parallelism"**; provably faster than SD & non-SD, lossless. | No — theoretical; does **not** cost inter-device comm | Latency; multi-instance |
| **PEARL** — Parallel SD w/ adaptive draft length (ICLR 2025; **2408.11850**) | 2 concurrent instances; pre-verify + post-verify overlap draft & target (kills mutual-waiting). | No | Latency; single/multi-GPU |
| **SpecExec** — Svirschevski et al. (NeurIPS 2024; **2406.02532**) | Consumer multi-GPU + **RAM offloading**; up to 20 tokens/target-pass via big draft trees. | Offload-IO-aware, not network | Latency; single node offloaded |
| **CoSine — Collaborative Speculative Inference** (2025; **2503.10325**) | **Multi-node**: decouple drafting/verification, route to specialized drafters, confidence fusion. | Claims cluster comm optimization | Throughput; multi-node cluster |
| **SpecEdge** — Edge-assisted serving (2025; **2505.17052**) | **Disagg (edge↔server, WAN):** edge drafts, server verifies, exchange **tokens only**. | **Yes** (minimizes network payload) | Cost/throughput; geo-distributed |
| **SpecInfer** — (ASPLOS 2024; **2305.09781**) | **TP/multi-GPU** + offloading; reports 1.5–2.8× distributed; tokens-only comm. | Reports distributed speedup, no explicit model | Latency/throughput; multi-GPU |
| **DSD — Decentralized Spec Decoding** (2025; **2511.11733**) | Verifies candidates in parallel across distributed nodes; **"turns communication latency into computation throughput."** Reduces cross-node comm ≈ `(N−1)·t1·(k−1)/k`. Lossless. **Does NOT address MoE/EP.** | **Yes** (dense, decentralized/WAN) | Throughput; multi-node |
| **DistServe** — (OSDI 2024; **2401.09670**) | Prefill/decode **disaggregation** baseline (not SD); models KV-transfer comm. Context for disagg+SD. | Yes (KV transfer) | Throughput/goodput; multi-node |
| **Helix** — (ASPLOS 2025; **2406.01566**) | Max-flow/MILP placement over heterogeneous GPUs+network (not SD). Context for comm-modeling. | Yes (network in max-flow) | Throughput/latency; multi-node |

**vLLM implementation note (docs.vllm.ai):** SD works with target-model TP, but EAGLE drafts run at
`draft_tensor_parallel_size=1`, and **SD is currently incompatible with pipeline parallelism** — the
small-batch-TP / draft-TP-mismatch pain that EasySpec and SwiftSpec target is real in production.

**Critical question — is distributed SD genuinely comm-/EP-/multi-node-aware, or unexplored?**

Partially explored on three axes, but the project's exact target — **comm-bound multi-node MoE
(expert-parallel, all-to-all) SD** — is essentially **white space**:

- **Pipeline parallelism is the most developed distributed direction.** PipeInfer, FlowSpec, SpecPipe
  operate across stages/clusters; PipeInfer explicitly optimizes low-bandwidth interconnects.
- **Disaggregation + TP is emerging and gives the strongest comm-aware example.** SwiftSpec explicitly
  names small-batch-TP communication overhead and disaggregates to fix it; SpecEdge minimizes WAN
  payload; CoSine pipelines draft/verify across nodes. But all **dense**, mostly single-node-8-GPU.
- **Expert parallelism / MoE is the clear gap.** Every MoE-SD paper (Cascade 2506.20675, MoESD
  2505.19645, MoE-SpeQ 2511.14102, SP-MoE, SpecMoEOff, EVICT 2605.00342, MoE-Spec 2602.16052) frames
  the "union of experts" blow-up as a **single-node memory-bandwidth / PCIe-offload** cost. **None
  model it as datacenter-scale all-to-all dispatch/combine traffic across EP ranks and nodes.** No
  verified paper co-designs the drafter or speculation length with the all-to-all collective, models
  NVLink-vs-InfiniBand cost of a K-token verify step under EP, or studies how draft-token expert
  fan-out changes all-to-all message volume across nodes. DSI is the only work framed as *distributed*
  SD but costs no inter-device comm; DSD (2511.11733) is comm-aware but dense-only.

---

## 7. Synthesis

### (a) Consensus on SD at serving batch — does it help, and where is the crossover?

- **Default (short context):** SD is a bandwidth-for-compute arbitrage that pays off at **low batch**
  and turns **net-negative past a crossover around batch ~32** as decode becomes compute-bound
  (Synergy 2310.18813). Serving systems should **adaptively shrink or disable** speculation under high
  load (SmartSpec 2406.14066: gains vanish by request rate ~12, hurt by ~16; larger models cross over
  sooner). Below ~0.5 draft acceptance, SD loses at any batch.
- **Modern refinement:** the crossover is about **arithmetic intensity, not batch per se**. Weight-ops
  amortize with batch (become compute-bound) but **KV-cache ops scale with `batch × seqlen` and stay
  memory-bound**. So at **long context + large batch, decode stays memory-bound and SD helps at
  serving scale** (MagicDec 2408.11049: up to 2.51× at batch 32–256, and speedup *grows* with batch).
  There is effectively a critical sequence length above which the memory-bound window reopens at any batch.
- **MoE twist (central for this project):** genuinely regime-dependent and the two key papers differ in
  emphasis. **MoESD (2505.19645):** at moderate batch, MoE benefits *more* than dense, and *sparser*
  MoE *widens* the useful batch range (expert FFNs are memory-bound with spare compute → verifying
  draft tokens is nearly free). **Cascade / Utility-Driven MoE (2506.20675):** the opposite failure —
  a draft batch activates the *union* of experts, inflating weight movement/verify time 2–3× and
  potentially slowing MoE by 1.5×. **Net: whether SD helps a MoE hinges on how many extra experts the
  draft batch activates per verify step — itself batch- and sparsity-dependent — so a utility/goodput-
  gated, batch-adaptive speculation length is the converged design.**

### (b) Is a cheaper VERIFY losslessly possible, or accepted as irreducible?

- The **accepted-token verify forward is treated as irreducible** under exact SD: no method lowers a
  committed token's own forward FLOPs/comm below one full target pass while provably preserving the
  target distribution.
- **Losslessly reducible are:** (i) the *number* of verified tokens/branches (SpecDec++, EVICT,
  Cascade); (ii) *wasted* verify work — reject doomed drafts early with periodic re-validation (HiSpec)
  or prune rejected tokens mid-verify (FASER); (iii) in **MoE**, the *redundant expert loading and the
  surrounding data movement* — tree-branch pruning to shrink the expert union (EVICT), and
  prefetch/overlap to *hide* offload/transfer (SP-MoE, SpecMoEOff, MoE-SpeQ).
- **Lossy** the moment you touch the accepted forward itself: cap experts (MoE-Spec), partial KV
  (SpecPV), or sparsify attention/FFN/MoE at verify (2512.21911).
- **Open gap:** a *lossless* reduction of the **intrinsic per-token MoE verify compute or all-to-all
  communication** (e.g., verify a committed token with a provably-sufficient subset of experts, or with
  reduced all-to-all volume, and get identical output) **does not exist**. Token-tree verification is
  the lossless "verify many candidates in one forward" primitive — and MoE's expert-union cost is
  exactly what *breaks* its cheapness, which is where the opportunity lies.

### (c) Gaps relevant to a cheaper DRAFT or cheaper VERIFY — comm-aware / serving-batch-positive

1. **EP/all-to-all-aware SD is white space.** The MoE-SD literature models the draft-tree expert-union
   blow-up as a *local memory-bandwidth* cost; **nobody models or exploits it as multi-node all-to-all
   dispatch/combine traffic.** A drafter or speculation-length policy co-designed with the all-to-all
   collective (e.g., biasing drafts toward tokens whose experts are already resident on-rank, or
   bounding the expert fan-out of a verify batch to cap all-to-all message volume) is unexplored.
   *This is precisely the comm-bound multi-node MoE regime the project targets.* Anchors to build from:
   **SwiftSpec** (comm-aware SD+TP, dense), **MoESD** (MoE-SD batch-regime model), **EVICT** (lossless
   MoE tree truncation to shrink the expert union), and **Semantic Parallelism / "Speculative MoE"**
   (2503.04398, lossless token↔expert co-scheduling to cut all-to-all — but not SD).

2. **Serving-batch-positive SD needs the MoE + long-context intersection.** MagicDec shows KV keeps
   decode memory-bound at large batch for long context; MoESD shows sparser MoE widens the useful batch
   range. **Their intersection — long-context, high-batch, sparse-MoE, multi-node — is where SD is most
   likely to remain throughput-positive, yet is unstudied jointly.** A drafter that is cheap in the
   *communication* sense (activates few/local experts, exchanges only tokens) rather than only cheap in
   FLOPs is the natural lever.

3. **Lossless per-token MoE verify reduction is unclaimed.** All lossless MoE work reduces *how many*
   tokens/branches or *hides* transfer; capping the *intrinsic* expert set of a committed token is
   currently only done lossily (MoE-Spec, 2512.21911). A losslessness argument for verifying accepted
   tokens with a reduced expert/all-to-all footprint would be novel.

4. **Cheaper DRAFT that is comm-aware, not just FLOP-cheap.** Self-draft heads (EAGLE-3, DeepSeek-MTP)
   and self-spec (LayerSkip, Kangaroo) minimize *draft FLOPs/params*; none minimize *draft communication*
   under EP/TP. Under expert parallelism a draft step still triggers all-to-all. A drafter whose routing
   is **local/on-rank by construction** (no or minimal all-to-all during drafting) is an unaddressed
   design point directly aligned with the project's "local/EP-aware draft" direction.

---

## Appendix — verified arXiv IDs (quick index)

**Foundations:** Blockwise Parallel 1811.03115 · Leviathan 2211.17192 · Chen/DeepMind 2302.01318 · Survey (Xia) 2401.07851
**Self-draft heads:** Medusa 2401.10774 · Hydra 2402.05109 · EAGLE 2401.15077 · EAGLE-2 2406.16858 · EAGLE-3 2503.01840 · Meta-MTP 2404.19737 · DeepSeek-V3 2412.19437 · ReDrafter 2403.09919 · Falcon 2412.12639
**Tree/structured:** SpecInfer 2305.09781 · Sequoia 2402.12374 · OPT-Tree 2406.17276 · DySpec 2410.11744 · ProPD 2402.13485 · RSD 2402.14160 · SMART 2604.09731 · Traversal Verification 2505.12398 · Batch-SD-Done-Right 2510.22876
**Batching/throughput:** Synergy 2310.18813 · MagicDec 2408.11049 · SmartSpec/TurboSpec 2406.14066 · SpecDec++ 2405.19715 · DISCO 2405.04304 · BASS 2404.15778 · TriForce 2404.11912 · Ouroboros 2402.13720 · Sarathi-Serve 2403.02310 · MoESD 2505.19645 · Cascade/Utility-MoE 2506.20675
**Verify-cost:** Staged 2308.04623 · Cascade Drafting 2312.11462 · Draft&Verify 2309.08168 · LayerSkip 2404.16710 · Kangaroo 2404.18911 · SWIFT 2410.06916 · Lookahead 2402.02057 · REST 2311.08252 · Faster Cascades 2405.19261 · HiSpec 2510.01336 · FASER 2604.20503 · SpecPV 2512.02337 · EVICT 2605.00342 · MoE-Spec 2602.16052 · SP-MoE 2510.10302 · SpecMoE 2604.10152 · SpecMoEOff 2508.21706 · MoE-SpeQ 2511.14102 · Sparse-Verify 2512.21911
**Distributed/EP:** SwiftSpec 2506.11309 · EasySpec 2502.02493 · Llama-at-Scale 2508.08192 · PipeInfer 2407.11798 · FlowSpec 2507.02620 · SpecPipe 2504.04104 · DSI 2405.14105 · PEARL 2408.11850 · SpecExec 2406.02532 · CoSine 2503.10325 · SpecEdge 2505.17052 · DSD-Decentralized 2511.11733 · DistServe 2401.09670 · Helix 2406.01566 · Semantic-Parallelism/"Speculative MoE" 2503.04398 *(adjacent, not SD)*

*Caveats:* IDs `26xx.*` are recent 2026 preprints. The roofline blog numbers are illustrative, not
peer-reviewed. Semantic Parallelism (2503.04398) is an EP-communication scheduler, not a draft/verify
SD method. Falcon (2412.12639) is a spec-decode method, distinct from the Falcon LLM family.
