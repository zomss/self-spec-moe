# Literature Review: MoE LLM Inference Acceleration

**Scope.** Current (2023–2026) work on accelerating Mixture-of-Experts (MoE) LLM
*inference*, framed for a systems project on **speculative decoding for
communication-bound multi-node MoE serving**. Every method below was verified
against a real title + arXiv id / venue via web search (July 2026). Papers with
`YYMM` ids in the 2506–2606 range are 2025-H2 through 2026-H1 preprints confirmed
live on arXiv.

## 0. Framing: the three cost axes and the MoE decode balance

A dense-transformer decode step is dominated by **weight-read memory-I/O**
(arithmetic intensity ∝ batch size). MoE breaks this: each token touches only
`k` of `E` experts, so the **effective arithmetic intensity of an expert GEMM
scales with `batch × (k/E)`** and therefore **stays memory-bound even at large
serving batch** (DeepSeek-V3 hardware note, arXiv:2505.09343; MegaScale-Infer,
arXiv:2504.02263). Expert Parallelism (EP) shards experts across GPUs/nodes and
adds a **sparse, data-dependent all-to-all** (dispatch + combine) per layer,
which becomes the dominant cost in wide-EP decode:

- In a MoE layer, inter-device communication can occupy **~47% of execution
  time** (COMET, arXiv:2502.19811).
- vLLM's production wide-EP DeepSeek trace: "the MoE Dispatch/Combine section
  shows the outsize duration spent in collective communication, **despite the
  small compute load**" (vLLM Large-Scale Serving blog, 2025-12-17; 2.2k tok/s/H200,
  up from ~1.5k).

So MoE decode at serving batch is simultaneously **weight-memory-bound** (per
expert) *and* **communication-bound** (all-to-all), while compute (FLOPs) is
cheap. This is the axis structure every method below trades against. Note the
axes for each method: **COMPUTE** (FLOPs), **MEM-I/O** (weight/KV reads),
**COMM** (a2a/EP), and whether it helps at **high (serving) batch** or only low
batch.

---

## 1. Expert-parallel all-to-all reduction / comm–compute overlap

State of the art in making the MoE all-to-all cheap or hidden. Two families:
**(a) overlap** the a2a with compute (hide it), and **(b) shrink** the a2a
(fewer bytes / fewer hops). DeepEP is the off-the-shelf kernel substrate; COMET,
FLUX, TokenWeave, DBO are the overlap schedulers; Tutel/Lina/Janus/FasterMoE/
ScheMoE/MegaBlocks are the (mostly training-era) foundations.

| Method (year, venue/id) | Mechanism | COMPUTE | MEM-I/O | COMM | Batch |
|---|---|---|---|---|---|
| **DeepEP** (2025, DeepSeek OSS; used in vLLM/SGLang) | Device-initiated (NVSHMEM + IBGDA) sparse a2a dispatch/combine kernels; **FP8 dispatch**; HT kernels (prefill) + LL kernels (decode) with near-zero SM occupation | – | – (frees SMs) | **Shrinks + hides**: FP8 payload ≈½ bytes; LL kernels cut decode a2a latency; overlap-friendly | Both; LL kernels target low-latency decode |
| **COMET** (2025, MLSys/ByteDance prod, arXiv:2502.19811) | Data-dependency analysis + task rescheduling for *fine-grained* overlap of dispatch/combine with expert GEMMs; adaptive workload assignment | neutral | – | **Hides** (1.96× single-layer, 1.71× e2e) | High (deployed at 10k-GPU scale) |
| **FLUX** (2024, ByteDance, arXiv:2406.06858) | Over-decomposes GEMM+comm into *tiles*, fuses comm into GEMM kernel epilogue/prologue (TP and EP) | neutral | – | **Hides** at tile granularity | High |
| **TokenWeave** (2026, MLSys'26, arXiv:2505.11329) | Wave-aware token-splitting into 2 microbatches; fuses AllReduce+RMSNorm; separate compute/comm streams. Beats Flux/NanoFlow/TileLink | neutral | – | **Hides** (up to 29% latency / 26% throughput; sometimes beats "comm removed") | Works down to 1024 tokens (TP; MoE-adjacent) |
| **DBO — Dual Batch Overlap** (2025, vLLM/DeepSeek DualPipe-style) | Split batch into 2 microbatches on 2 threads/2 streams; ping-pong so one does a2a while the other computes (`--enable-dbo`) | **Doubles** exposed compute (halves FLOPs/s per µbatch) | – | **Hides** (drives exposed a2a ≈0) | **High only**; *hurts* at small batch (memory-bound, compute penalty outweighs) |
| **MegaScale-Infer** (2025, SIGCOMM'25, arXiv:2504.02263) | Disaggregate attention vs FFN modules; **ping-pong pipeline parallelism** shuttles microbatches attn↔FFN; custom M2N comm lib (no GPU↔CPU copies) | neutral | Raises expert GEMM batch → better util | **Hides** + tailored M2N | High |
| **Tutel** (2023, MLSys, arXiv:2206.03382) | Adaptive parallelism/pipelining at zero switching cost; 2D-hierarchical (2DH) all-to-all; fast encode/decode | – | – | **Shrinks hops** (hierarchical a2a); 4.96–5.75× layer | Both (training-origin) |
| **Lina** (2023, ATC, arXiv:2210.17223) | Prioritize a2a over allreduce; tensor partitioning + pipelined micro-op scheduling for a2a | – | – | **Hides + schedules** (1.73× train, 1.63× p95 infer) | Both |
| **FasterMoE** (2022, PPoPP) | Roofline model; **dynamic shadowing** (pull hot expert weights instead of sending tokens); topology-aware expert selection | – | Moves weights not tokens | **Shrinks/reroutes** congestion | Training |
| **Janus** (2023, SIGCOMM, 3603269.3604869) | **Data-centric**: keep tokens in place, *fetch experts* (async, hierarchical, intra-node sharing) instead of a2a on tokens | – | ↑ (experts fetched/cached) | **Replaces a2a** when experts < data | Training |
| **ScheMoE** (2024, EuroSys, 3627703.3650083) | Modularize compress/comm/expert-compute; optimal adaptive pipeline schedule; pluggable a2a + **compression before a2a** | – | – | **Shrinks + hides** | Training |
| **MegaBlocks** (2023, MLSys, arXiv:2211.15841) | **Dropless** MoE via block-sparse GEMM kernels — no token dropping, no capacity-factor waste/padding | Removes padding waste; +40% vs Tutel | – | orthogonal | Both |
| **MegaScale-MoE** (2025, arXiv:2505.11432) | Production training; replaces BF16 reduce-scatter with **FP8 all-to-all** (FP32 reduction) | – | – | **Shrinks** (½ bytes) | Training |
| **NCCL-EP** (2026, arXiv:2603.13606); **Triton-distributed** (2025, arXiv:2504.19442); **NVSHMEM analysis** (2026, arXiv:2606.05951) | Portable device-initiated EP a2a APIs / compilers / measurement studies generalizing DeepEP-style primitives beyond DeepSeek's stack | – | – | **Hides/shrinks** infra | Both |

**Token dropping vs dropless.** Capacity-factor token-dropping caps a2a volume
and pads GEMMs (quality loss); MegaBlocks/dropless (block-sparse) removes the
tradeoff and is now standard in inference stacks. Modern serving (vLLM/SGLang)
is dropless.

**SOTA answer.** For *hiding* comm: DeepEP kernels + a microbatch overlap
scheduler (DBO in vLLM, COMET/ping-pong in production) is the state of the art;
DeepEP is **off-the-shelf and lossless**. For *shrinking* comm: FP8/INT8
dispatch (DeepEP, DeepSeek-V3, MegaScale-MoE) is deployed and near-lossless.

**Is any of it spec-decode-aware? No.** All of §1 assumes one token per
sequence per step. None model the fact that a speculative step verifies `γ+1`
tokens per sequence — which multiplies the *unique experts touched* and thus the
dispatch fan-out (see §5). This is the central gap.

---

## 2. MoE quantization for inference

Weights dominate MoE parameter budget, so weight-only quant gives the biggest
**MEM-I/O** win; activation quant additionally **shrinks the a2a payload**
(links §2 → §1). Effect on FLOPs is via low-precision tensor cores (FP8/FP4).

| Method (year, id) | Mechanism | COMPUTE | MEM-I/O | COMM | Batch |
|---|---|---|---|---|---|
| **gpt-oss MXFP4** (2025, OpenAI) | MoE **weights in MXFP4** → 120B fits one 80GB GPU | FP4 tensor cores | **↓↓** (4-bit experts) | – | Both |
| **MxMoE** (2025, arXiv:2505.05799) | Accuracy–perf co-design: per-expert mixed bitwidth via joint optimization; custom mixed-precision Group-GEMM kernels | ↓ | **↓↓** | – | Both |
| **EAQuant** (2026, arXiv:2506.13329) | Expert-aware PTQ handling routing-induced calibration imbalance | – | **↓↓** | – | Both |
| **MoEQuant** (2025, arXiv:2505.03804) | Expert-balanced sampling + affinity guidance for MoE PTQ | – | **↓↓** | – | Both |
| **MoPEQ** (2025, arXiv:2509.02512) | Per-expert precision by activation frequency (hot experts higher-bit) | – | **↓↓** | – | Both |
| **Dynamic Expert Quant** (2025, arXiv:2511.15015) | Runtime bitwidth per expert under routing imbalance / memory traffic | – | **↓↓** (scalable) | – | Both |
| **EAC-MoE** (2025, arXiv:2508.01625) | Expert-Selection-Aware Compressor (quant + expert pruning aligned to routing) | ↓ | **↓↓** | – | Both |
| **Efficient MoE Quant w/ guarantees** (2026, arXiv:2604.06515) | Expert-wise mixed precision: higher bits for small router-norm experts; theoretical generalization bound | – | **↓↓** | – | Both |
| **FP4 MoE training on Hopper** (2026, arXiv:2603.02731) | Direct FP8→FP4 (de)quant; scaling-aware row↔col conversion; **FP4 activations + FP4 expert-parallel comm** | FP4 GEMM | ↓ | **Shrinks a2a >50%** (FP4 activations) | High (prefill-heavy) |
| **NVFP4 vs MXFP4 sensitivity** (2026, arXiv:2603.08747) | Layer/block-wise sensitivity of the two FP4 formats; NVFP4 (16-elem blocks) > MXFP4 accuracy on Blackwell | FP4 | ↓ | – | Both |
| **HOBBIT** (2025, arXiv:2411.01433) | Mixed-precision **expert offloading**: load low-bit version of cold experts to cut PCIe I/O | – | **↓↓** (offload I/O) | – | Low (offload) |
| **MoE + Mixture-of-Precisions for QoS** (2024, arXiv:2407.14417) | Serve different expert precisions to hit latency SLOs | ↓ | ↓ | – | Both |
| **Bandwidth-Efficient Adaptive MoE (low-rank comp.)** (2025, arXiv:2512.17073) | Low-rank compensation for aggressively quantized experts to restore quality | – | **↓↓** | – | Both |

**Key cross-axis point.** Activation quant is the only lever in §2 that touches
**COMM**: FP8 dispatch (DeepSeek-V3 quantizes *dispatch only*, keeps combine
BF16) and FP4 activations (arXiv:2603.02731) cut a2a payload ~2–4×. This is the
one place quantization and the comm axis meet — relevant to making a spec-decode
draft's dispatch cheaper.

---

## 3. Expert offloading / caching / selective loading

Single-/few-GPU regime: keep hot experts resident, fetch cold experts from
CPU/DRAM/SSD on demand. All about **MEM-I/O + locality**; irrelevant to
multi-node COMM (no EP a2a); helps mainly at **low batch** (where sparse
activation and temporal locality survive — at high batch nearly all experts
activate every step, killing the cache).

| Method (year, id) | Mechanism | COMPUTE | MEM-I/O | COMM | Batch |
|---|---|---|---|---|---|
| **Mixtral-offloading** (2023, arXiv:2312.17238) | **LRU expert cache** + quant to run Mixtral-8×7B on consumer GPUs | – | prefetch + cache | – | Low |
| **Fiddler** (2024, arXiv:2402.07033) | CPU–GPU orchestration: **compute cold experts on CPU** rather than move weights (win when batch small); 90GB model >3 tok/s on 24GB GPU | shifts to CPU | avoids weight transfer | – | Low |
| **MoE-Infinity** (2024, arXiv:2401.14361) | Sequence-level Expert Activation Matrices → activation-aware prefetch + LFU cache; 4–20× latency, >8× cost | – | **↓↓** (temporal locality) | – | Low |
| **Pre-gated MoE** (2024, ISCA, arXiv:2308.12066) | Algorithm-system co-design: a **pre-gate** predicts next layer's experts one layer early → lookahead prefetch + expert-aware batching | – | **hides fetch** | – | Low |
| **SiDA-MoE** (2024, MLSys, arXiv:2310.18859) | Offline-trained **hash function** replaces router to predict activation → data-aware prefetch; 3.93× throughput, latency to 28% | – | **↓↓** | – | Low |
| **EdgeMoE** (2023, arXiv:2308.14352) | On-device engine: expert-specific bitwidth + compute-I/O pipeline preload; evicts distant-layer experts first | ↓ (bitwidth) | **↓↓** | – | Low (edge) |
| **ExpertFlow** (2024, arXiv:2410.17954) | Predictive routing + token scheduling to co-optimize expert activation and loading | – | **↓↓** | – | Low |
| **DAOP** (2025, DATE, arXiv:2501.10375) | Data-aware offload + **predictive pre-calculation** of experts | overlaps compute | **↓↓** | – | Low |
| **Fate** (2025, arXiv:2502.12224) | **Cross-layer gate**: predict experts from *previous* layer's inputs for accurate prefetch | – | **↓↓** | – | Low |
| **PreScope** (2025, arXiv:2509.23638) | Prefetching tuned for resource-constrained MoE (overlap SSD/CPU I/O) | – | **↓↓** | – | Low |
| **FlashMoE** (2026, arXiv:2601.17063) | **ML-based cache replacement** blending recency+frequency for SSD I/O; +51% hit rate vs LRU/LFU, up to 2.6× | – | **↓↓** | – | Low (edge/SSD) |
| **OD-MoE** (2025, arXiv:2512.03927) | On-demand expert loading for **cacheless** edge-distributed MoE | – | ↓ (no cache) | – | Low |
| **CoX-MoE** (2026, arXiv:2605.17889) | Coalesced expert execution with **AMX** CPU–GPU co-execution for throughput | CPU AMX | ↓ | – | Mid |
| **Context-aware MoE on CXL** (2025, arXiv:2512.04476) | Expert placement across **CXL-attached GPU-NDP** memory tiers | near-data | **↓↓** | – | Low/Mid |

**Locality story / caveat.** Offloading exploits per-sequence temporal locality
of expert activation. It is a **low-batch / single-node** technique and is
largely *orthogonal* to multi-node EP comm — but several spec-decode-for-MoE
papers (§5) fuse offloading with speculation, and this is where offloading
resurfaces for our problem.

---

## 4. MoE-specific serving systems & the comm-vs-compute balance

Papers that *characterize* when MoE decode is comm-bound vs compute-bound, and
disaggregated / wide-EP serving systems.

| Work (year, id) | Contribution to the balance question |
|---|---|
| **DeepSeek-V3 tech report** (2024, arXiv:2412.19437) & **HW insights** (2025, arXiv:2505.09343) | Canonical wide-EP serving recipe: **PD-disaggregation**, EP32 prefill / **EP≈256 + DP** decode, 32 redundant experts + EPLB load balancer. States expert arithmetic intensity ∝ batch×activation-ratio → **stays memory-bound at large batch**. |
| **MegaScale-Infer** (2025, SIGCOMM'25, arXiv:2504.02263) | Argues sparse activation shifts FFN **from compute- to memory-intensive**, lowering GPU util → motivates attn/FFN disaggregation + ping-pong PP. |
| **Revealing Challenges of Attn–FFN Disaggregation** (2026, arXiv:2602.09721) | Extends the **roofline to the communication level** (bandwidth × arithmetic intensity × HFU); finds a "**dead zone**" where adding FFN instances fails to raise utilization because scale-out bandwidth caps the workload — a direct comm-bound diagnosis. |
| **Fine-Grained Scheduling of Disaggregated EP** (2025, arXiv:2512.21487) | Scheduler for disaggregated expert parallelism decode. |
| **Rethinking Network Topologies for Cost-Effective MoE Serving** (2026, arXiv:2605.00254) | Co-designs interconnect topology to the a2a pattern; quantifies DBO benefit vs batch (helps at large batch, hurts small). |
| **Survey on Inference Optimization for MoE** (2024, arXiv:2412.14219) | Taxonomy across compute/memory/comm; good anchor. |
| **LMSYS large-scale EP blog** (2025) / **vLLM wide-EP blog** (2025-12) | Production evidence: DeepSeek on 96×H100 PD-disagg + large-scale EP; vLLM **2.2k tok/s/H200**; explicitly names dispatch/combine as the dominant decode cost despite tiny compute. |

**Verdict on the balance.** The field now agrees: **wide-EP MoE decode is
communication-bound** (dispatch/combine dominate; compute is small; per-expert
memory reads are the secondary wall). Roofline-to-comm analyses (2602.09721)
formalize a comm-limited "dead zone." This is exactly the regime a spec-decode
project targets.

---

## 5. MoE + speculative decoding

The most relevant sub-area. Central tension (well established now): **naive spec
decode is anti-synergistic with MoE** because verifying `γ+1` draft tokens
collectively **activates far more unique experts** than a single token — raising
both **MEM-I/O** (weight reads) and **COMM** (dispatch fan-out) and inflating
verification cost, unlike dense models where verification is ~free.

| Method (year, id) | Mechanism & key finding | COMPUTE | MEM-I/O | COMM |
|---|---|---|---|---|
| **MoESD** (2025, NeurIPS'25 spotlight, arXiv:2505.19645) | Analysis: **at medium batch, MoE benefits *more* from SD than dense**; sparser MoE → *broader* effective batch window. New "target efficiency" metric. Up to **2.29× (Qwen2-57B-A14B)** | analysis | key: draft tokens raise expert activation | analysis (single-node) |
| **Utility-Driven SD for MoE / "Cascade"** (2025, arXiv:2506.20675) | Finds draft tokens **increase data movement + verification time by 2–3×**, causing up to **1.5× slowdown**. "Speculation utility = token-gain / verify-cost"; dynamic-K test phases. **Caps slowdown to 5%, +7–14% throughput** | – | **quantifies the MEM-I/O blowup** | notes verify comm growth |
| **MoE-Spec** (2026, arXiv:2602.16052) | **Verification-time expert budgeting** (training-free): load only top-contribution experts, drop the long tail. Verification cost is **non-constant, scales with #unique experts**. **+10–30% throughput vs EAGLE-3** at equal quality | ↓ | **↓** (fewer experts loaded) | **↓** (fewer dispatched — closest to comm-aware) |
| **SP-MoE** (2025, arXiv:2510.10302) | Use draft-model **attention states to predict + prefetch the target's experts** before verification (structural draft↔target correspondence); cutoff-layer prefetch depth; async pipeline. **1.07–3.5× TPOT** | – | **↓↓** (prefetch hides fetch) | offload-bandwidth focus |
| **MoE-SpeQ** (2025, arXiv:2511.14102) | Speculative **quantized** decoding + proactive expert prefetch/offload for memory-limited MoE | ↓ (quant) | **↓↓** | – |
| **ELMoE-3D** (2026, arXiv:2604.14626) | **Elastic Self-Speculative Decoding**: exploit MoE's expert-count *and* bit-width elasticity; a truncated-expert/low-bit self-draft doubles as expert cache; HW/SW co-design on 3D-stacked (hybrid-bonding) memory. **6.6× / 4.4× energy at batch 1–16** | ↓ | **↓↓** (HB bandwidth) | on-package (not multi-node) |
| **MoE-SpAc** (2026, arXiv:2603.09983) | Speculative **activation-utility** expert selection for heterogeneous edge | ↓ | ↓ | – |
| **Jakiro** (2025, arXiv:2502.06282) | **MoE *as the draft head*** — decoupled multi-head draft via MoE for diverse candidates (accelerates *dense* targets; MoE is the mechanism, not the target) | – | – | – |
| **Speculative MoE / "Semantic Parallelism"** (2025→ICLR'26, arXiv:2503.04398) | *"Speculative" pre-scheduling, not spec-decoding*: predict co-activated experts + tokens and **co-locate them on the same device** (speculative token shuffling + expert grouping) to **cut a2a**. Output lossless (falls back to normal a2a on mispredict); comm savings best-effort | – | – | **Shrinks a2a** (fewer remote hops) — but for *ordinary* decode, not for a spec-decode verify step |

**What the SD+MoE literature found about comm.** Almost nothing directly.
The dominant finding is a **MEM-I/O** result: draft tokens blow up expert
activation / weight reads (MoESD, Cascade, MoE-Spec). The systems responses are
either **verification-time expert budgeting** (MoE-Spec — reduces both weight
reads *and*, incidentally, dispatch count) or **prefetch/offload** (SP-MoE,
MoE-SpeQ, ELMoE-3D) — all evaluated **single-GPU or offloading**, none on a
**multi-node EP all-to-all**. "Speculative MoE" uses the word "speculative" for
expert *pre-scheduling* (a comm-reduction for *normal* decode), not for
speculative *decoding*.

---

## 6. Synthesis

### (a) SOTA for reducing MoE all-to-all comm — lossless/off-the-shelf vs research idea

- **Off-the-shelf & lossless (deployed today):**
  - **DeepEP** device-initiated dispatch/combine kernels (NVSHMEM/IBGDA), with
    separate **high-throughput** (prefill) and **low-latency** (decode) variants
    and **FP8 dispatch** — the de-facto substrate in vLLM/SGLang. Lossless
    (FP8 dispatch is near-lossless; combine stays BF16).
  - **Microbatch overlap** to *hide* the a2a behind compute: **DBO** in vLLM,
    **ping-pong PP** in MegaScale-Infer, production **COMET**. All lossless.
    Caveat: overlap needs a **large batch** to pay off; at small/decode batch
    the split's compute penalty can erase the win (DBO explicitly hurts small
    batch).
  - **Payload shrinking via activation quant**: FP8 dispatch (DeepSeek-V3),
    FP8 a2a (MegaScale-MoE), FP4 activations/comm (arXiv:2603.02731, >50%
    traffic cut) — deployed, near-lossless.
  - **Systemic**: PD-disaggregation + wide-EP + EPLB redundant-expert load
    balancing (DeepSeek-V3 recipe) — production standard.

- **Research ideas (not yet turnkey):** kernel-fusion tile overlap (FLUX,
  TokenWeave, TileLink), data-centric expert-fetch instead of token a2a
  (Janus/FasterMoE shadowing), co-scheduling tokens+experts to cut hops
  ("Speculative MoE"/Semantic Parallelism), comm-aware topology co-design
  (arXiv:2605.00254), portable EP APIs (NCCL-EP, Triton-distributed).

**Bottom line:** the comm-*hiding* SOTA (DeepEP + microbatch overlap) is mature,
lossless, and off-the-shelf — **but it is optimized for a single token per
sequence per step and needs large batch to hide comm.**

### (b) The gap for spec-decode DRAFT / VERIFY on the comm axis

Two research communities do not intersect:

1. **Comm/overlap systems (§1, §4)** are **not spec-decode-aware.** They model
   one token/sequence/step. A speculative **verify** step processes `γ+1`
   tokens/sequence, which (i) **multiplies the number of unique experts** each
   rank must receive, inflating dispatch fan-out and the a2a message count, and
   (ii) changes the compute:comm ratio of the microbatch — so DBO/COMET-style
   overlap tuning and DeepEP LL-vs-HT kernel choice are **mis-calibrated** for a
   verify step. No paper re-derives the overlap/kernel-selection schedule for
   `γ+1`-token verification.

2. **Spec-decode-for-MoE systems (§5)** are **not comm-aware.** MoESD, Cascade
   (2506.20675), MoE-Spec (2602.16052), SP-MoE, MoE-SpeQ, ELMoE-3D all analyze
   and optimize the **MEM-I/O / weight-read** blowup on **single-GPU or
   offloading** hardware. **None** measure or optimize the EP **all-to-all** of
   a verify step in a **multi-node** deployment. MoE-Spec's verification-time
   expert budgeting is the *closest* — pruning verify experts implicitly cuts
   dispatch fan-out — but it is framed as a memory optimization and evaluated
   single-node, never as an a2a reduction.

**Concrete open gaps a spec-decode project can own:**
- **Comm cost model of a MoE verify step**: how `γ`, acceptance rate, and
  batch jointly set dispatch fan-out and whether the step is comm- or
  memory-bound (extend the roofline-to-comm of arXiv:2602.09721 to spec decode).
- **Draft-side expert prediction to pre-shape the verify a2a** (fuse SP-MoE's
  draft-state expert prediction with "Speculative MoE"-style token/expert
  co-location so the verify dispatch stays local) — untried.
- **Spec-decode-aware DBO / DeepEP LL-vs-HT selection** and expert-budgeted
  verification (MoE-Spec) *specifically to bound multi-node a2a volume*, not
  just weight reads.
- **Self-spec via expert-subset draft (SS-MoE-style)** that makes the *draft*
  a low-fan-out (few-expert / low-EP) pass so the draft's a2a is cheap and the
  verify's a2a is predictable — the elasticity idea of ELMoE-3D exists only for
  on-package hybrid-bonding memory, **not for multi-node EP comm.**

In short: **the comm axis of speculative MoE decode is essentially unstudied at
multi-node scale.** The tools to attack it (DeepEP, DBO, expert budgeting, draft
expert prediction, expert-subset self-spec) all exist separately but have never
been combined against the all-to-all cost of a speculative verify.
