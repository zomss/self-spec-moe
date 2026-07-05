# Literature Review: Self-Speculative Decoding & Cheap-Draft Methods

**Scope.** Current (2023–2026) work on speculative decoding where the *draft* is produced from the target model itself or at minimal extra cost. Organizing question: **how do you make the DRAFT cheap on the three cost axes — COMPUTE, MEMORY-I/O, COMMUNICATION?** This is Direction A of the project (cheap draft, *not* MoE-specific). The COMMUNICATION axis is the project's distinctive angle (comm-bound multi-node MoE serving), so it is treated in the most depth.

All arxiv ids below were verified against real abstract pages / repos during this review. Where a method has no arxiv paper (e.g. prompt-lookup) this is stated explicitly.

---

## 0. Framing: the three cost axes of a draft

A verification step is (mostly) fixed by the target model. The draft is the tunable part. Its per-token cost decomposes as:

- **COMPUTE** — FLOPs the drafter spends per proposed token (matmuls in the draft path, tree width, LM-head over the vocab).
- **MEMORY-I/O** — bytes moved GPU-DRAM↔SRAM per draft token: draft weights loaded, draft KV cache loaded/stored, LM-head weight. In the decode regime this is the dominant latency term for small batches.
- **COMMUNICATION** — bytes over interconnect *per draft token* in a distributed deployment: TP all-reduce inside each layer, EP all-to-all in MoE, PP activations across stages, or draft↔target logit transmission in a disaggregated/collaborative setup.

The universal tension: making the draft cheaper on any axis lowers its fidelity, which lowers **acceptance length** τ (mean accepted tokens per verify). Net speedup ≈ τ / (1 + draft-cost-ratio). A "free" draft that accepts nothing is worthless; the art is a favorable point on the cost–acceptance curve.

---

## 1. Layer-skip / early-exit self-draft

Draft = a *sub-network* of the target (early layers, or a subset of layers). Verify = the full stack. Because draft and target **share weights and KV cache**, this family is inherently cheap on **memory-I/O** (no second model resident) and moderately cheap on **compute** (fewer layers). Its Achilles heel is acceptance.

| Method | id / venue | Mechanism | Axis reduced | Acceptance | Training? |
|---|---|---|---|---|---|
| **Draft & Verify** (Zhang et al. 2023) | `2309.08168`, ACL 2024 | Skip a selected set of intermediate layers (Bayesian-optimized skip set) to draft; full model verifies in one pass. Plug-and-play, zero extra params/memory. | compute + mem-I/O (fewer layers, shared weights/KV) | up to **1.99×** on LLaMA-2; lossless output | **No** |
| **LayerSkip** (Meta) | `2404.16710`, ACL 2024 | *Train* with layer-dropout + early-exit loss so all layers share one LM head; at inference exit early to draft, remaining layers verify. Draft & verify share compute + activations. | compute + mem-I/O; "less memory footprint than other SD" | 1.34–2.16× depending on task | **Yes** (special pretrain/finetune recipe) |
| **Kangaroo** | `2404.18911`, NeurIPS 2024 | Fixed shallow sub-network as self-draft + a tiny trained adapter (1 attn + 2 norms, 67M) to close the representation gap; *double* early-exit halts drafting when confidence drops. | compute + mem-I/O; 88.7% fewer added params than Medusa | up to **1.68×** on Spec-Bench | **Yes** (adapter only, lightweight) |
| **SWIFT** | `2410.06916`, ICLR 2025 | *On-the-fly, training-free* layer-skip: an optimization phase searches the skip set for the live input stream, then an acceleration phase runs it. | compute + mem-I/O | >**1.6×**, distribution-preserving | **No** |
| **EESD / Early-exiting + Thompson Sampling** | `2406.03853` | Early-exit head after first N layers, improved by self-distillation; Thompson sampling adapts draft length. | compute + mem-I/O | strong on 13B/70B; lossless | **Yes** (self-distillation) |
| **CLaSp** | `2505.24196`, ACL 2025 | In-context layer-skip chosen by a **dynamic-programming** optimizer using the last verification step's hidden states as the objective; updates skip schedule each step. | compute + mem-I/O | training-free, adaptive; higher accept than static skip | **No** |
| **PPSD** | `2509.19368` | Configure layers as a **pipeline**: while final layers verify token *t*, the early-exit path drafts *t+1* ("verify-while-draft"), so no work is wasted on rejects. | compute (hides draft latency via PP) | **2.01–3.81×**, near-optimal at fixed accept rate | No (uses an early-exit model) |
| *(also)* SpecEE `2504.08850`, KnapSpec `2602.20217`, ConfLayers `2604.14612` | 2025–26 | Predictor-gated / knapsack / confidence-based layer-skip variants. | compute + mem-I/O | incremental | mixed |

**On the Phase-17 super-linear acceptance drop — is it field consensus? Yes, increasingly.**
- **"The Diminishing Returns of Early-Exit Decoding in Modern LLMs"** (`2603.23701`, 2026) is the clearest statement: early-exit "scores show a decreasing trend as LLMs evolve" — improved pretraining recipes and architectures **reduce layer redundancy**, shrinking the maximum layer-skip ratio that preserves behavior. Dense transformers retain more early-exit headroom than **MoE and SSMs**; base models > instruction-tuned; >20B > smaller. This is direct external support that layer-skip acceptance degrades faster than the compute you save, and that MoE targets are *especially* poor for layer-skip drafts — relevant to our MoE setting.
- The self-speculative-via-layer-skip papers themselves (Draft&Verify, SWIFT, CLaSp) all spend their engineering on *choosing which layers to skip* precisely because naive uniform skipping collapses acceptance — implicitly the same finding.
- Practical consequence: layer-skip is cheap on compute + mem-I/O but **acceptance-limited**, and gets worse on newer/MoE models. It does **nothing** for the communication axis (a skipped layer still all-reduces / all-to-alls the layers it keeps).

---

## 2. Quantized self-draft

Draft = a **low-precision copy of the same weights** (and/or a quantized KV cache). Attacks **memory-I/O** directly: 4-bit weights/KV move 4× fewer bytes per token. Losslessness argument: the quantized draft only *proposes*; the **full-precision target verifies**, so the output distribution is exactly the target's (rejection sampling), regardless of draft quantization error — quantization only costs acceptance, never correctness.

| Method | id / venue | Mechanism | Axis reduced | Acceptance | Training? |
|---|---|---|---|---|---|
| **QuantSpec** | `2502.10424` (Apple/UCB) | Self-draft = same architecture with **4-bit weights + hierarchical 4-bit quantized KV cache**; full-precision verify. Targets the KV-load bottleneck in long context. | **mem-I/O** (4-bit weights+KV); ~1.3× less memory than sparse-KV self-SD | **>90%** accept; up to **2.5×** e2e | No |
| **QSpec** | `2410.11305`, EMNLP 2025 | Two complementary quant schemes: low-precision **activation-weight joint** quant to draft fast, **high-precision weight-only** quant to verify; **reuses the same weights + KV** → near-zero-cost switching. | mem-I/O + compute; zero extra model | up to **1.64×** (1.55× batched), no quality loss | **No** |
| **Speculative Decoding Meets Quantization** | `2505.22179` | Finds tree-draft verification is *slower* than a single fwd pass on 4-bit models (compute, not memory, dominates once weights are 4-bit); fixes with a **hierarchical** small-model→sequence-draft stage. | mem-I/O w/ compute-aware fix | **2.78×** on 4-bit Llama-3-70B, 1.31× over EAGLE-2 | mixed |
| **SPEQ** | `2510.18525` | **Bit-sharing floating point (BSFP)**: extract a 4-bit draft from the *top bits* of the full FP16/BF16 weights → draft is literally a slice of the target, no extra storage or training. | mem-I/O; **zero extra memory** | **2.07×** vs FP16 | **No** |
| **ML-SpecQD** | `2503.13565` | Multi-level SD with **MXFP4 quantized drafts** as the first-level drafter, training-free, composable with EAGLE. | mem-I/O | training-free speedups | No |

**Takeaway.** Quantized self-draft is the cleanest **memory-I/O** win with a rigorous losslessness story. Acceptance stays high (>90% for QuantSpec) because a 4-bit copy is a *much* better approximation of the target than an early-exit sub-network. But it does not touch **communication** (a 4-bit weight still participates in the all-reduce; you move fewer bytes to SRAM, not fewer bytes over NVLink/IB) — though 4-bit *activations*, if the collective is on activations, would shrink comm volume too (unexploited angle, see §5).

---

## 3. Draft-model-free / retrieval / n-gram (zero-compute draft)

Draft = looked-up tokens, no neural draft forward pass at all. **Compute and mem-I/O for drafting ≈ 0** (a trie/hash/suffix-tree lookup). Acceptance depends entirely on how repetitive the workload is; verification cost is unchanged.

| Method | id / venue | Mechanism | Axis reduced | Accept length / speedup | Training? |
|---|---|---|---|---|---|
| **Prompt Lookup Decoding** (Saxena 2023) | *no arxiv* — GitHub `apoorvumang/prompt-lookup-decoding`; in vLLM & HF Transformers | Match last k tokens against the prompt, propose the continuation. | **all three ≈ 0** for drafting | **2–4×** on input-grounded tasks (summarization, QA, code-edit); ~1× otherwise | **No** |
| **REST** | `2311.08252`, NAACL 2024 | Retrieve from a **datastore**, build a trie of continuations, prune low-freq branches, draft from trie. | compute+mem-I/O of drafting ≈ 0 (adds datastore memory) | **1.62–2.36×** (7B/13B) | **No** |
| **Lookahead decoding** | `2402.02057`, ICML 2024 | No draft model: **Jacobi** fixed-point iteration generates + verifies n-grams in parallel; trades log(FLOPs)/step for fewer steps. | *raises* compute/step to cut step count; no draft weights/KV | linear step reduction w/ FLOPs; ~1.5–2× | **No** |
| **SuffixDecoding** | `2411.04975`, NeurIPS 2025 Spotlight | **Suffix trees** over prompt + prior outputs; adaptive speculation depth by empirical token freq. Built for agentic/repetitive workloads. | compute+mem-I/O of drafting ≈ 0 | up to **5.3×** on agentic (SWE-Bench, text-to-SQL); 2.8× over EAGLE-2/3 | **No** |
| **Token Recycling** | `2408.08696`, ACL 2025 | Store candidate tokens (incl. rejected) in an **adjacency matrix**, BFS a draft tree, verify with tree attention; ~2 MB state. | compute+mem-I/O of drafting ≈ 0 | ~**2×**, general-purpose (no repetition assumption) | **No** |

**When do they help?** Only when the output *copies* from context (RAG, code edit, agentic loops, multi-turn). On open-ended generation, accept length collapses to ~1 and they add nothing. They are the *only* family that is trivially cheap on **all three** axes for the drafting step — but they buy that with a workload assumption, not a general fidelity guarantee. **Crucially for us: because the draft is pure lookup, it never triggers any collective communication.** This is the one existing family whose draft is intrinsically communication-free (see synthesis).

---

## 4. Small trained heads attached to the target (self-draft)

Draft = a few extra parameters bolted onto the target's last hidden state. Adds a little **compute + memory** (head weights + head KV) but reuses the target's forward pass features; no separate model.

| Method | id / venue | Mechanism | Head cost | Speedup | Training? |
|---|---|---|---|---|---|
| **Medusa** | `2401.10774` | K extra MLP heads predict tokens t+1..t+K from the last hidden state; tree attention verifies. | K small MLP heads (~0.5–0.6B for Vicuna-7B config); no head KV | ~2.3–2.8× | **Yes** (Medusa-1 frozen backbone; Medusa-2 joint) |
| **EAGLE** | `2401.15077`, ICML 2024 | Autoregress at the **feature** (2nd-to-top) level with one lightweight transformer layer; reuse target LM head; tree draft. | one transformer layer + its KV | ~2.7–3× | **Yes** (draft layer) |
| **EAGLE-2** | `2406.16858`, EMNLP 2024 | Dynamic, confidence-expanded/pruned draft tree on the EAGLE head. | same head | up to ~3.5–4× | Yes (reuse EAGLE head) |
| **EAGLE-3** | `2503.01840`, NeurIPS 2025 | Multi-layer feature fusion + "training-time test"; drops the feature-prediction constraint, scales with data. | one head, richer features | ~4×+ | **Yes** |
| **Multi-Token Prediction (MTP)** | `2404.19737` (Gloeckle/Meta) | Train n parallel output heads on a shared trunk; heads double as a self-draft at inference. | n heads on shared trunk | modest at inference; better base model | **Yes** (pretrain-time) |
| **DeepSeek-V3 MTP** | `2412.19437` | **Single-layer** MTP module (output head shared with main model) predicts the next 2 tokens → drop-in self-draft. | 1 transformer layer | **1.8× TPS**; **85–90%** accept of 2nd token | **Yes** (part of pretraining) |

**Cost of the head itself.** The head is cheap in FLOPs but adds **memory-I/O** (its weights + its own small KV cache must be loaded each draft step) and, in a distributed target, the head's matmuls **also communicate** if they are sharded (an EAGLE transformer layer has the same TP all-reduce as any layer; the LM head is vocab-parallel). MTP/EAGLE give the best acceptance in this survey (feature-level drafting is far more faithful than layer-skip), which is why they dominate leaderboards — but they need training and are **not** communication-free.

---

## 5. The COMMUNICATION axis of the draft (the project's angle — sparse but non-empty)

This is where the field is thinnest. Almost all §1–4 work assumes the draft runs on one GPU (or replicates the small draft), so *communication* never enters the draft cost model. Where communication *does* appear, it splits into three distinct sub-threads:

### 5a. TP + speculative decoding is *incompatible at small batch* — systems fixes
- **SwiftSpec** (`2506.11309`, 2025) states the core problem plainly: "conventional approaches fail to apply both speculative decoding and tensor parallelism simultaneously due to imbalanced compute (draft vs target), KV-cache inconsistencies, and **communication overheads under small-batch tensor-parallelism**." Fix = **asynchronous, disaggregated** pipeline (parallel tree generation, tree-aware KV, fused kernels) so draft cost leaves the critical path. **1.75×** avg; Llama3-70B @ 348 tok/s on 8×H100. → *Reorganizes* the pipeline; does **not** make the draft's own communication cheaper.
- **EasySpec** (`2502.02493`, NeurIPS 2025): the draft's optimal TP size is *smaller* than the target's, so GPUs idle during drafting. Fix = **layer-parallel "fuzzy" drafting** — break inter-layer dependencies so draft layers run concurrently across the idle GPUs, then recalibrate the draft KV in one pass. **4.17×** peak, training-free. → Uses spare GPUs; the draft still all-reduces, just overlapped.
- **PPSD** (`2509.19368`): pipeline-parallel overlap of early-exit draft and verify (also in §1). → Hides draft latency behind PP, a communication-*aware* schedule.

### 5b. Disaggregated / collaborative (draft and target on *different* devices) — logit-transmission bandwidth
Here "communication" = sending the draft's output over a **network link** to the target device. This is the closest existing analogue to a "communication-cheap draft," and the fixes are about shrinking the transmitted distribution:
- **Communication-Efficient Collaborative LLM Inference via Distributed Speculative Decoding** (`2509.04576`, Zheng & Yang): baseline transmits **full-vocabulary logits every step** — the bottleneck. Fix = **Top-K Sparse Logits Transmission (TK-SLT)**: send only top-K probs + indices; derive the optimal draft length for throughput. → Directly a *communication-cheap draft*, but for the edge/collaborative topology, not intra-model collectives.
- **VocabTrim** (`2506.22694`): rebuild the drafter LM head to hold only frequently-sampled tokens → smaller LM-head matmul and smaller logit vector. **16%** memory-bound speedup on Llama-3.2-3B, **training-free**; slightly lower accept. → Shrinks both mem-I/O of the head *and* the size of any logit that must be moved/all-gathered.
- **SpecVocab / "Speculative Decoding with a Speculative Vocabulary"** (`2602.13836`, 2026): **dynamically** pick a per-step vocabulary subset (vs a fixed reduced vocab) to shrink the embedding/LM-head bottleneck; **+8.1%** throughput over EAGLE-3, higher accept than fixed-vocab. → Primarily compute, but the same lever (smaller vocab → smaller vocab-parallel all-gather) is a latent comm win.
- Related edge/collaborative SD: `2511.01695` (resource-aware parallel SD), `2512.09963` (GoodSpeed adaptive SD edge), ScienceDirect "Fast collaborative inference via distributed SD."

### 5c. Speculation used *to reduce* MoE communication (opposite direction, adjacent)
Not "cheap-comm draft" but "draft-to-cut-comm," worth knowing because it's the same MoE-comm battleground:
- **Speculative MoE / Sem-MoE** (`2503.04398`): predict which experts tokens will hit and **pre-schedule tokens+experts to collocate them**, cutting EP **all-to-all**. Uses speculation to minimize comm.
- **SP-MoE** (`2510.10302`): SD-aware **expert prefetching** + compute-comm pipelining.
- **MoE-SpeQ** (`2511.14102`): on-device draft predicts the *expert sequence* to **prefetch experts from host** (overlaps I/O).
- **MoE-Spec** (`2602.16052`): verification-time **expert budgeting** (fixed expert capacity) to decouple speculation depth from memory/comm cost.
- **Utility-Driven SD for MoE** (`2506.20675`): manages the extra experts a wide draft tree activates.

**Verdict on the communication axis.** There is real work, but it is either (i) systems reorganization to *tolerate* TP+SD (SwiftSpec, EasySpec, PPSD), (ii) logit-transmission compression for the *disaggregated/edge* topology where draft↔target is a network hop (TK-SLT, VocabTrim, SpecVocab), or (iii) speculation used *to reduce* MoE comm. **No paper found makes the self-draft avoid the target's own intra-model collectives** — i.e., draft by computing only what's *local* to each TP/EP rank (local argmax over the vocab shard, or drafting from only the local experts) so the draft step **skips the all-reduce / all-to-all entirely** and pays the collective only at verify. That specific "communication-avoiding self-draft" is the open slot (§7).

---

## 6. KV / memory-I/O reduction for the draft

Overlaps §2 but distinct: keep full-precision weights, make the *draft's KV footprint* small (sparse or shared). This is the dominant lever for **long-context** self-draft, where KV-load, not weight-load, is the bottleneck.

| Method | id / venue | Mechanism | Axis | Result |
|---|---|---|---|---|
| **TriForce** | `2404.11912` | Hierarchical: draft = **original weights + sparse (retrieved) KV**, itself speculated by a tiny StreamingLLM-cache model. | mem-I/O (KV) | **2.31×**, uses only **3% of KV** vs 68% | No |
| **MagicDec** | `2408.11049` | Draft with a **sparse/fixed-size KV** cache whose size doesn't grow with context → breaks the latency-throughput tradeoff at long context / large batch. | mem-I/O (KV) | up to **2.51×** at large batch, long seq | No |
| **QuantSpec** | `2502.10424` | 4-bit **quantized** KV as draft (see §2). | mem-I/O (KV bytes) | >90% accept, 2.5× | No |
| **LongSpec** | `2502.17421` | Constant-memory draft KV + anchor-offset positions + efficient tree verify for long context. | mem-I/O (KV) | lossless long-context speedup | Yes (drafter) |

**Shared KV between draft and target** is precisely what self-speculative (§1) and quantized-self-draft (§2) give for free: the draft reuses the target's prefix KV, so no second KV cache is built for the accepted prefix — the memory-I/O saving that makes self-draft attractive vs a separate small model.

---

## 7. Synthesis

### (a) Which axis does each family attack — and is anything cheap on ALL THREE?

| Family | COMPUTE | MEMORY-I/O | COMMUNICATION | Training | Accept quality |
|---|:---:|:---:|:---:|:---:|:---:|
| Layer-skip / early-exit (§1) | ✅ fewer layers | ✅ shared weights/KV | ❌ kept layers still collective | mixed | ⚠️ low, degrading on new/MoE models |
| Quantized self-draft (§2) | ~ | ✅✅ 4-bit weights+KV | ❌ (unless activations quantized) | mostly No | ✅ high (>90%) |
| Draft-free / retrieval / n-gram (§3) | ✅✅ ~0 | ✅✅ ~0 | ✅✅ **no draft collective at all** | No | ⚠️ workload-dependent |
| Trained heads: Medusa/EAGLE/MTP (§4) | ~ small head | ~ head weights+KV | ❌ head shards all-reduce | **Yes** | ✅✅ best (feature-level) |
| TP/EP systems fixes (§5a) | — | — | ⚠️ *tolerate* comm, not remove | mixed | n/a |
| Logit-transmission compression (§5b) | ~ | ✅ smaller LM head | ✅ smaller draft→target payload | mostly No | slight accept loss |
| Sparse/shared draft KV (§6) | — | ✅✅ KV bytes | ❌ | mixed | ✅ high |

**Only the draft-free / n-gram family (§3) is cheap on all three axes simultaneously** — because it does no neural draft forward pass, it emits no activations to reduce, gather, or route. But it pays with a **repetitiveness assumption**: acceptance is high only when the output copies context (RAG, agentic, code). No *general-purpose, fidelity-guaranteeing* draft is cheap on all three at once. Quantized self-draft is cheapest on memory-I/O with the best acceptance-per-cost; layer-skip is cheapest to deploy but acceptance-limited (and worsening on modern/MoE targets); trained heads win acceptance but need training and still communicate.

### (b) Is the COMMUNICATION axis of the draft essentially UNEXPLORED? — Largely YES, with three caveats.

The hypothesis holds in the specific sense that matters to us: **no work makes a *self*-draft that avoids the target's own TP all-reduce / EP all-to-all.** The near-neighbors are all *different problems*:
1. **Systems reorganization** (SwiftSpec/EasySpec/PPSD) makes TP+SD *coexist* — it overlaps or disaggregates draft communication, it does not eliminate it. The draft still pays every collective.
2. **Logit-transmission compression** (TK-SLT/VocabTrim/SpecVocab) is genuinely "communication-cheap draft," but only for the **draft-on-a-separate-device** topology (edge/collaborative), where comm is a network hop, not the intra-layer collective inside a single tensor-/expert-parallel target.
3. **Speculation-to-reduce-MoE-comm** (Speculative MoE, SP-MoE, MoE-SpeQ) uses drafting to *schedule* experts — the inverse of making the draft itself comm-cheap.

So: the intersection **{self-draft} × {avoid the target's collective communication} × {multi-node TP/EP MoE}** appears **empty**. The comm axis of the draft is under-theorized: most cheap-draft papers don't even include communication in their cost model.

### (c) Gaps → openings for a genuinely communication-cheap draft

1. **Communication-avoiding self-draft (the flagship gap).** Draft using only rank-local computation so the draft step **skips the collective**: e.g. per-TP-rank **local argmax over the vocab shard** / partial-sum logits (draft from an *approximate* de-synchronized state, all-reduce only at verify), or draft from a **layer/attention approximation that is block-diagonal across ranks**. Pay the all-reduce/all-to-all once per *verified block*, not once per draft token. This directly amortizes the collective over τ, converting the comm-bound regime into a compute/mem regime. **No prior work does this** — the closest (EasySpec's "fuzzy" layer-parallel draft with periodic KV recalibration) validates the *pattern* of "draft from a deliberately de-synchronized state, resync at verify," but does it for GPU-utilization, not to skip collectives.
2. **EP-local self-draft for MoE.** In multi-node MoE, drafting is comm-bound by the router's **all-to-all**. A draft that routes only to **experts already resident on the local node** (top-k restricted to local experts, or a cheap local-expert approximation) would **skip the all-to-all** for the draft, paying it only at verify. This is the MoE-specialized instance of gap 1 and is the project's Direction-A×MoE sweet spot. Existing MoE-SD work prefetches or schedules experts; none *draft from local experts to avoid the collective*. Note the §1 finding that MoE targets are *bad* for layer-skip drafts — so an EP-local draft (not a layer-skip draft) is the right primitive for MoE.
3. **Comm-aware acceptance model.** The field's cost model is `speedup ≈ τ/(1+c)` with c = compute/mem ratio. In multi-node MoE, **c is dominated by communication**, and it is *per-draft-token*. Re-deriving the optimal draft length / tree width under a **communication-weighted** cost (à la TK-SLT's optimal-draft-length derivation, but for intra-model collectives) is unaddressed and would quantify the payoff of amortizing collectives over a block.
4. **Compose the axes.** A draft that is quantized (4-bit *activations* → also 4× smaller all-reduce payload, §2) **and** EP-local (§gap 2) **and** shares KV (§6) could be cheap on all three axes *with* a fidelity guarantee — the combination no single paper has assembled.

**One-line gap statement.** Cheap-draft research has thoroughly mined compute (layer-skip, n-gram) and memory-I/O (quantized/sparse-KV self-draft), lightly touched draft↔target *network* comm in the disaggregated/edge setting, but has **not** built a self-draft that avoids the target's own TP/EP collectives in a single multi-node deployment — the exact lever for comm-bound MoE serving.

---

## Appendix: verified reference list (arxiv id — title — venue)

- `2309.08168` — Draft & Verify: Lossless LLM Acceleration via Self-Speculative Decoding — ACL 2024
- `2404.16710` — LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding — Meta, ACL 2024
- `2404.18911` — Kangaroo: Lossless Self-Speculative Decoding via Double Early Exiting — NeurIPS 2024
- `2410.06916` — SWIFT: On-the-Fly Self-Speculative Decoding — ICLR 2025
- `2406.03853` — Speculative Decoding via Early-exiting w/ Thompson Sampling (EESD) — 2024
- `2505.24196` — CLaSp: In-Context Layer Skip for Self-Speculative Decoding — ACL 2025
- `2509.19368` — PPSD: Pipeline Parallelism … Early-Exit Self-Speculative Decoding — 2025
- `2603.23701` — The Diminishing Returns of Early-Exit Decoding in Modern LLMs — 2026
- `2504.08850` — SpecEE: Speculative Early Exiting — 2025 (adjacent)
- `2502.10424` — QuantSpec: Self-Speculative Decoding w/ Hierarchical Quantized KV Cache — Apple/UCB 2025
- `2410.11305` — QSpec: Speculative Decoding with Complementary Quantization Schemes — EMNLP 2025
- `2505.22179` — Speculative Decoding Meets Quantization — 2025
- `2510.18525` — SPEQ / "From Quarter to All": Bit-Sharing FP 4-bit self-draft — 2025
- `2503.13565` — ML-SpecQD: Multi-Level Speculative Decoding with Quantized Drafts — 2025
- `2402.02057` — Break the Sequential Dependency … Lookahead Decoding — ICML 2024
- (no arxiv) — Prompt Lookup Decoding — GitHub apoorvumang/prompt-lookup-decoding, 2023 (in vLLM/HF)
- `2311.08252` — REST: Retrieval-Based Speculative Decoding — NAACL 2024
- `2411.04975` — SuffixDecoding: Model-Free / Extreme Speculative Decoding — NeurIPS 2025 Spotlight
- `2408.08696` — Token Recycling (Turning Trash into Treasure) — ACL 2025
- `2401.10774` — Medusa: Multiple Decoding Heads — 2024
- `2401.15077` — EAGLE: Speculative Sampling / Feature Uncertainty — ICML 2024
- `2406.16858` — EAGLE-2: Dynamic Draft Trees — EMNLP 2024
- `2503.01840` — EAGLE-3: Training-Time Test scaling — NeurIPS 2025
- `2404.19737` — Better & Faster LLMs via Multi-Token Prediction (MTP) — Meta 2024
- `2412.19437` — DeepSeek-V3 Technical Report (single-layer MTP self-draft) — 2024
- `2404.11912` — TriForce: Hierarchical SD, sparse-KV draft — 2024
- `2408.11049` — MagicDec: Breaking Latency-Throughput Tradeoff (sparse-KV draft) — 2024
- `2502.17421` — LongSpec: Long-Context Lossless Speculative Decoding — 2025
- `2506.11309` — SwiftSpec: Async Speculative Decoding under TP — 2025 **(comm)**
- `2502.02493` — EasySpec: Layer-Parallel Speculative Decoding, multi-GPU — NeurIPS 2025 **(comm)**
- `2509.04576` — Communication-Efficient Collaborative LLM Inference via Distributed SD (TK-SLT) — 2025 **(comm)**
- `2506.22694` — VocabTrim: Vocabulary Pruning for Efficient Speculative Decoding — 2025 **(comm/mem)**
- `2602.13836` — SpecVocab: Speculative Decoding with a Speculative Vocabulary — 2026 **(comm/compute)**
- `2503.04398` — Speculative MoE / Sem-MoE: comm-efficient parallel MoE inference — 2025 **(comm, inverse)**
- `2510.10302` — SP-MoE: SD + expert prefetching for MoE — 2025 **(MoE)**
- `2511.14102` — MoE-SpeQ: Speculative Quantized Decoding w/ expert prefetch — 2025 **(MoE)**
- `2602.16052` — MoE-Spec: Expert Budgeting for Efficient Speculative Decoding — 2026 **(MoE)**
- `2506.20675` — Utility-Driven Speculative Decoding for MoE — 2025 **(MoE)**
