# Paper outline: Communication-Aware Speculative Decoding for MoE Serving

Working title (pick one):
- **"Speculative Decoding is Communication-Bound on Mixture-of-Experts: Comm-Free Drafts and Comm-Aware Trees"**
- "Communication-Aware Speculative Decoding for MoE Expert-Parallel Serving"

One-line thesis: *On MoE served with expert parallelism off NVLink, the decode all-to-all
dominates; because the verifier processes all proposed tokens in one forward, its
communication scales with the number of verified tokens -- which reshapes both how to
draft (communication-free) and how to size the verify tree (small/pruned).*

---

## Abstract (draft)

MoE models are increasingly served with expert parallelism (EP), where each decode step
performs an all-to-all to route tokens to experts. Off NVLink -- over PCIe within a node or
across nodes over the network -- this all-to-all dominates: serving is **communication-
bound**. We show this regime reshapes speculative decoding. Because the verifier processes
all proposed (chain/tree) tokens in one forward, its all-to-all **scales with the number of
verified tokens**. Two consequences follow. **(1)** The draft should avoid the all-to-all,
and need not be trained: a communication-free local-routing pass over a small resident FP4
expert cache (kept small by a verify-warmed dynamic cache + skip-cold) yields a lossless,
**training-free self-speculative decoder (World A)** -- ~1.2-1.35x on single-node PCIe with
no draft head. **(2)** The verify tree must be small: the wide trees used by EAGLE/MTP route
~5-15x the tokens to accept ~4, inflating the verify all-to-all; a **communication-aware
small/pruned tree (World B)** is **~2.3x** faster than the default wide tree on MoE-EP. We
characterize the design space -- acceptance, memory, and the speedup ceilings -- across
three MoE families, including negatives (tree pruning and cascaded verification cannot beat
a chain; draft-overlap favors cheap drafts except deep in the comm-bound regime). The
benefits grow inter-node, where communication dominates most.

---

## 1. Introduction

- MoE + EP serving is the norm for large models; decode is memory/comm-bound.
- Speculative decoding accelerates decode but is studied in compute-bound (single-GPU,
  NVLink) settings where the verify is ~free per extra token.
- **Gap:** off NVLink, MoE-EP decode is communication-bound; the verify all-to-all is the
  cost, and it scales with the number of verified tokens. Existing spec-decode design
  choices (wide trees; "make the draft comm-free") need to be re-examined.
- **Contributions (crisp):**
  - **C1 (analysis):** spec decoding on MoE-EP is comm-bound; the verify all-to-all scales
    with verified tokens and with batch; quantify the machine balance and the f curve.
  - **C2 (World B, main systems result):** comm-aware tree-spec. Wide trees (EAGLE/MTP
    default) blow up the verify all-to-all; small/pruned trees give ~2.3x. Draft-agnostic.
  - **C3 (World A):** a training-free, lossless comm-free self-speculative decoder, with a
    memory mechanism (verify-warmed dynamic / globally-hot cache + skip-cold) to keep the
    resident draft experts small.
  - **C4 (design space + negatives):** acceptance/memory characterization across 3 MoE
    families; honest negatives and regime splits.

## 2. Background & motivation

- MoE-EP decode step: gate -> dispatch all-to-all -> experts -> combine all-to-all, per layer.
- Machine balance: FFN arithmetic intensity vs interconnect; NVLink ~compute-bound, PCIe /
  inter-node ~comm-bound. **[Phase 18 framing, Phase 24 measured]**
- Speculative decoding + rejection sampling = lossless; tree/EAGLE/MTP drafting.
- The measurement testbed: forcing the EP all-to-all over real PCIe on an NVLink box
  (`NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=1` -> SHM; the NVLS-disable
  insight). vLLM EP = NCCL all_gatherv/reduce_scatterv. **[Phase 24 3e]**

## 3. The communication-bound analysis (C1)

- Measured comm fraction f on forced-PCIe MoE-EP: f = 0.37-0.65 (B=8->512), saturating
  ~0.62; verify 23/27/35/48 ms vs NVLink 15-17. **[Phase 24 3e/3g; Fig 1]**
- **Key result:** the verify is one forward over all proposed tokens -> comm = N_coll*L
  (fixed latency, amortized) + (tokens*payload)/BW (scales with verified tokens). So spec's
  verify communication scales with k (chain depth) / tree width. **[Phase 27]**
- Implication preview: small structures (chains/small trees), comm-free drafts; the rest of
  the paper.
- Inter-node projection: f=0.6-0.8 even at low batch (network hop). **[Phase 20 injection]**

## 4. World B: communication-aware tree-spec (C2 -- lead systems result)

- EAGLE2/3 / MTP draft WIDE trees (~25-64 nodes) and accept ~4 -> on MoE-EP the verify
  routes ~5-15x the tokens -> the all-to-all blows up.
- **Measured (forced-PCIe, B_seq=64):** verify step 40.9 / 47.0 / 110.8 ms for tree
  N=4 / 6 / 30; wide = 2.4x step, 4.7x bandwidth of the small tree. End-to-end:
  EAGLE-default wide **1.26x**, comm-aware pruned **2.61x**, comm-aware chain **2.93x**
  -> **~2.3x better**. **[Phase 33; Fig 2 (verify vs tree size), Fig 3 (speedup bars)]**
- Pruner fidelity: confidence-pruning is near-oracle; a wide tree prunes to 89% accept at
  5x fewer nodes. **[Phase 31 Stage A; Fig 4 recall-vs-compression]**
- Takeaway: on MoE-EP draft chains / small (pruned) trees; the chain is the comm-floor.

## 5. World A: training-free comm-free self-speculative decoding (C3)

- Draft = comm-free local routing (no all-to-all) over a resident FP4 expert cache; verify
  = full-EP bf16, exact (lossless). **[Phase 18, 24 B1]**
- **Acceptance levers (measured):** FP4 0.92 / FP8 0.95 weight quant **[Phase 22]**; FP8
  activations free on the wire **[Phase 23]**; shared-expert anchor +0.1..0.58
  **[Phase 25]**. Real multi-token accept matches one-step beta; lossless (rejection
  theorem; greedy-numerics caveat). **[Phase 24 B1, k-sweep]**
- **Memory mechanism (the enabler):** the resident draft experts must be small.
  - Static frontier: accept-length-per-GB ~0.32/GB to 0.5E, then a knee. **[Phase 26]**
  - Verify-warmed dynamic cache + verify-context-KV: last-committed-token cache -> beta
    0.99 at 1 GB (k=1). **[Phase 28; Fig 5]**
  - Batched serving: the resident set must cover the UNION of concurrent requests
    (grows to ~full by B~32) -> globally-hot top-C + skip-cold (verify corrects) -> 8 GB,
    beta ~0.91, batch-independent. **[Phase 28 union/skip-cold]**
- **Result:** ~1.2-1.35x single-node PCIe, lossless, training-free; memory 1-8 GB by
  regime. **[Phase 24 3f; Fig 6 speedup vs batch]**
- Cross-model: union growth + skew + shared anchor hold on DeepSeek-V2-Lite & GPT-OSS.
  **[Phase 29]**

## 6. Design space & negative results (C4 -- honest)

- **Tree pruning / cascaded verification do NOT beat a chain** (chain is the comm-floor;
  pruned-wide ~+2.5%, +7% oracle; cascade beats chain only if phase-1 ~free). **[Phase 31]**
- **Draft-overlap favors a CHEAP draft;** World A's full-forward draft only hides (and
  beats EAGLE) once comm > 2x compute (inter-node); a model, hardware-gated. **[Phase 32]**
- **Regime splits:** speedup wants high batch, memory win wants low batch; single-node PCIe
  is the milder cousin of inter-node. **[Phase 30]**
- **Ceilings:** comm-free draft depth capped by context degradation; verify-comm scaling
  caps k; single-node ceiling ~1.35x. **[Phase 28, 30]**

## 7. Evaluation (consolidated)

- Hardware: 8xH100 NVSwitch, forced-PCIe (NVLS-off). Models: Qwen3-30B-A3B, DeepSeek-V2-Lite,
  GPT-OSS-20B, Moonlight-16B.
- Figures: F1 f-vs-batch; F2 verify-vs-tree-size; F3 World B speedup bars; F4 prune
  recall-vs-compression; F5 dynamic-cache beta-vs-GB; F6 World A speedup-vs-batch;
  F7 cross-model union/skew; F8 (projected) inter-node f and speedups.
- Tables: acceptance-lever menu; memory strategies; negatives summary.

## 8. Related work

- Speculative decoding (Leviathan, Chen); EAGLE 1/2/3, Medusa, MTP, Sequoia, SpecInfer
  (tree drafting -- all compute-bound assumptions); cascade/staged spec; PipeSpec/async.
- MoE-EP serving + all-to-all (DeepEP, DBO); MoE quantization (NVFP4/MXFP4/FP8).
- **Our delta:** the comm-bound regime for spec on MoE-EP; verify-comm-scaling; comm-aware
  trees; training-free comm-free self-spec + its memory mechanism.

## 9. Limitations & future work

- Single-node PCIe measured; **inter-node IB is where the win is largest** (projected,
  hardware-gated). World A's draft-overlap and the World B gap both widen there.
- World B accept uses a local-routing **stand-in for EAGLE** (ratio robust since verify comm
  is draft-independent); a real EAGLE/MTP head would sharpen absolute numbers.
- No integrated distributed step-level driver yet (components measured, end-to-end composed).

## 10. Conclusion

On MoE-EP off NVLink, speculative decoding is communication-bound, and the verify's
all-to-all scales with verified tokens. This single fact yields a concrete systems win for
existing tree-spec (comm-aware small trees, ~2.3x over wide trees) and a training-free
lossless self-speculative decoder for when no draft head exists.

---

## Claims -> evidence map (for rigor / rebuttal)

| claim | status | phase |
| --- | --- | --- |
| MoE-EP decode is comm-bound off NVLink (f=0.37-0.65) | measured | 24 |
| verify all-to-all scales with verified tokens | measured | 24,27,33 |
| wide trees blow up verify comm (2.4x step/4.7x BW) | measured | 33 |
| comm-aware tree ~2.3x over EAGLE-default wide | measured (EAGLE accept = stand-in) | 33 |
| confidence-pruning near-oracle (89% accept, 5x fewer nodes) | measured | 31 |
| World A lossless, ~1.2-1.35x single-node | measured/composed | 24 |
| dynamic cache 1 GB/beta0.99 (k=1); 8 GB/beta0.91 batched | measured | 28 |
| acceptance levers (FP4/FP8/shared anchor) | measured, 3+ models | 22,23,25,29 |
| pruning/cascade cannot beat a chain | measured/modeled | 31 |
| overlap favors cheap draft; World A wins inter-node | modeled | 32 |
| inter-node f=0.6-0.8 -> larger wins | projected (injection) | 20 |

## What would make it camera-ready (the two gated experiments)
1. A **real EAGLE3/MTP head** on a supported MoE -> replace World B's stand-in accepts.
2. A **2-node IB** run -> measured inter-node f, World B gap, and World A's overlap.
