# Lossless Self-Speculative MoE Decoding on a Single PCIe Server

**Synthesis of Phases 24-31** (incl. the EAGLE/cascade survival test and a comparison to
conventional speculative decoding). Date: 2026-06-29. All numbers measured on 8xH100 (NVSwitch)
unless noted; model = Qwen3-30B-A3B (128 experts, top-8, no shared) unless noted;
cross-model = DeepSeek-V2-Lite, GPT-OSS-20B (Phase 29).

---

## 0. Thesis

MoE serving off NVLink is **communication-bound**: the expert all-to-all dominates the
decode step. A self-speculative scheme can hide that cost losslessly:

- **Draft** = comm-free, low-precision: local routing (no all-to-all) over a resident
  FP4 expert cache.
- **Verify** = bf16 full-EP, exact, with rejection sampling -> output distribution is the
  target model's, independent of the draft (**lossless**, Leviathan/Chen theorem).

The draft trades the expensive communication for cheap local compute; the verify pays the
communication once per accepted run. The question this program answers: **how much does it
buy on a single PCIe server, and what does the draft cost in memory and accept length?**

---

## 1. The speedup (Phase 24)

**Forcing real PCIe on an NVLink box.** `NCCL_P2P_DISABLE` alone does NOT leave NVLink:
NVLS (NVLink multicast over NVSwitch) is a separate path. The full recipe routes the EP
all-to-all over real PCIe (GPU<->host<->GPU SHM), no NVLink, no hang:

```
NCCL_P2P_DISABLE=1  NCCL_NVLS_ENABLE=0  NCCL_IB_DISABLE=1   ->  channels "via SHM/direct"
```

vLLM's EP all-to-all is plain NCCL (`cuda_communicator.all_gatherv -> pynccl`), so this
steers it. (The earlier "NCCL_P2P_DISABLE hangs" was NCCL falling back to NET->IB.)

**Measured (attention-DP8 + EP, real PCIe-SHM):**

| global B | NVLink step | PCIe-SHM verify | comm fraction f |
| ---: | ---: | ---: | ---: |
| 8 | 14.7 | 23.3 | 0.37 |
| 32 | 15.2 | 26.8 | 0.43 |
| 128 | 15.3 | 35.0 | 0.56 |
| 512 | 16.9 | 48.2 | 0.65 |

The comm-free draft step (skip the all-to-all) is 17.9-19.9 ms -- confirmed comm-free.

**The verify-cost correction (Phase 27 corrects Phase 24 3f).** The spec verify is ONE
forward over the k draft tokens, so its comm-bound all-to-all routes `B*k` tokens -> verify
cost scales with k. The naive composition (using the 1-token verify cost for a k-token
verify) was optimistic. With correct scaling, even *perfect* acceptance caps the high-batch
speedup at ~1.35x (k=1): the comm-free draft (~23 ms) nearly equals the comm it saves
(~25 ms). **Corrected single-node-PCIe lossless speedup: ~1.2-1.35x** (k=1 optimal at high
batch). Batch scaling saturates (f -> ~0.62 by B=1024; both comm and compute scale with
batch).

**This is a throughput-regime, not latency, effect, and it is modest.** The bigger win is
inter-node (IB is a network hop: Phase 20 f=0.6-0.8 -> 1.6-2.3x even at low batch) --
hardware-gated.

---

## 2. The draft's acceptance (Phases 25, 27)

The draft's per-token acceptance beta drives accept length. Levers (one-step beta, measured):

| lever | beta | note |
| --- | ---: | --- |
| local subset 0.5E (no quant) | ~0.85 | coverage-limited |
| FP4 (NVFP4) full coverage | 0.92 | Blackwell-native; MXFP4 0.905 |
| FP8 full coverage | 0.95 | 2x the memory of FP4 |
| + shared-expert anchor | +0.1..+0.58 | shared-expert models, EP-invariant |

**Shared anchor (Phase 25).** An always-local shared expert lifts local-routing beta a lot
(DeepSeek-V2-Lite +0.25..+0.58; Moonlight +0.13..+0.54; Qwen1.5 +0.09..+0.13), tracking the
shared mass fraction. It also buffers quant error. The realized comm-free FP4 draft on a
shared-expert model = ~0.82 @ 0.5E, ~0.92 @ full.

**Tree drafting (Phase 27).** Branching is very effective -- measured
`h(b)=P(verify-next in draft top-b)` = 0.875 / 0.979 / 1.000 for b=1/2/3: the draft's top-3
almost always contains verify's token. BUT the verify processes the whole tree, so verify
cost scales with node count (same comm penalty as large k). Optimal structure is batch-
dependent and SMALL:

| batch | best tree | speedup | vs chain |
| ---: | --- | ---: | ---: |
| 8 (latency) | full d2,b2 (6 nodes) | 1.21x | +8% |
| 128 / 512 | 1-token chain | 1.18-1.27x | +0% |

**Trees help only at low batch** (verify is latency-bound there -> node count ~free). At
serving batch the comm-bound verify penalizes nodes -> a 1-token chain wins.

Validated with real tree attention: accept lengths match the h(b)+geometric model within
2-4% (full d2b2 1.95 vs 1.94; +23% over chain); **lossless** (per-cycle fresh-reference
match 0.990 -- the lower global match is B1 batched-numerics cascade, not a bug);
**KV-correct** (draft KV is never reused at verify; verify recomputes full-weight). The
verify-context-KV draft (now default) adds only +0.017 beta standalone but is what enables
the memory result below.

---

## 3. The draft's memory (Phase 28)

The only extra memory is the resident FP4 draft experts. With the verify-context-KV draft
the context is full-weight, so the cache only needs the PREDICTING token's experts.

**Low batch (1 request/device): dynamic cache wins big.** depth-0 beta vs cache C:

| policy | C=8 (1 GB) | C=16 (2 GB) | C=32 (4 GB) | C=64 (8 GB) |
| --- | ---: | ---: | ---: | ---: |
| static prompt top-C | 0.45 | 0.57 | 0.77 | 0.90 |
| verify-warmed (EMA) | 0.64 | 0.89 | 0.99 | 0.99 |
| last-committed-token | **0.99** | 0.99 | 0.99 | 0.99 |

The predicting token is shaped by the LAST COMMITTED token's experts, which verify knows ->
caching exactly those (top_k=8 -> C=8) gives **beta 0.99 at 1 GB** (the "oracle" is
realizable, causal). 8x less than the static frontier's 8 GB for 0.90.

**Two hard limits (honest):**
1. **Depth cap.** The last-token cache is depth-0 only -- it collapses for k>1 (0.99 ->
   0.36 -> 0.21) because deeper draft tokens predict from UNVERIFIED draft tokens. For
   trees you need C=32 (4 GB, beta ~0.6-0.8 across depth). And **even the oracle decays with
   depth** -- the in-cycle draft tokens are local (no clean KV), so accumulating local
   context degrades deep predictions regardless of cache. A FUNDAMENTAL cap on comm-free
   draft depth.
2. **Batched-memory cap.** Experts are SHARED weights per DEVICE, not per request, so the
   resident set must cover the UNION of concurrent requests. union(B) per layer: 8 (B=1) ->
   41 (B=8) -> 77 (B=32) -> 103 (B=128) of 128. So the 1 GB per-request win is LOW-BATCH
   only; high batch needs ~the full set.

**High batch (serving): globally-hot + skip-cold, batch-independent.** A fixed globally-hot
top-C cache; cold-routed tokens draft their top resident experts + renorm; verify corrects
(lossless). Measured skip-cold beta (exceeds the coverage proxy -- renorm recovers the
dropped low-weight experts' mass):

| resident | FP4 mem | skip-cold beta |
| --- | ---: | ---: |
| globally-hot 0.25E | 4.1 GB | 0.71 |
| **globally-hot 0.5E** | **8.2 GB** | **0.94** (bf16) / ~0.91 (FP4) |
| full FP4 replication | 16.3 GB | 0.92 |

So **8 GB (0.5E) + skip-cold ~= full replication's beta at half the memory, batch-independent.**

---

## 4. Cross-model validation (Phase 29)

The enablers are not Qwen3 artifacts:
- **Union growth -> universal** (per-device union -> ~full routed set by B~32 on Qwen3,
  DeepSeek, GPT-OSS; faster for fewer-expert models).
- **Routing skew -> holds** (global-hot 0.5E cache covers Qwen3 0.92 / GPT-OSS 0.78 /
  DeepSeek-routed 0.70 -- all skewed, Qwen3 most).
- **Shared-expert anchor -> the differentiator**: DeepSeek's routed experts cache worst,
  but its shared experts (free, batch-independent, ~48% of MoE mass) cover half the output
  -> shared-expert architectures have a strictly more favorable memory story.

---

## 5. Does this survive a trained draft head? (Phase 31, EAGLE/MTP)

The honest stress-test: in a world that already has EAGLE3/MTP, does self-MoE-spec still
earn its keep? EAGLE's head is dense and comm-free, so it **subsumes the "comm-free draft"
thesis at the draft layer** (and, since it uses no MoE experts, it also moots the resident-
expert memory mechanism of Sec. 3). The only place a self-spec MoE forward could still add
value is the **exact-verify tree** (whose all-to-all scales with node count, Sec. 1/2).
Three variants tested, all NO-GO:

- **31a (free prune of our own wide tree).** Confidence-pruning is near-oracle (P_conf ~95%
  of P_exact), but a pruned-wide tree beats a plain CHAIN by only ~+2.5% accept at matched
  verify nodes (+7% even at oracle). Economically it loses: you pay the full wide-tree
  draft, and pruning only saves the verify. High batch 0.66-0.87x vs chain 1.18-1.27x;
  low batch merely ties Phase-27's small tree.
- **31b / two-phase cascade (EAGLE draft -> quantized comm-free phase-1 prune -> full
  phase-2 verify).** Valid and lossless, and it elegantly unifies the two worlds. But
  sweeping the phase-1 cheapness q (FP8-H100 ~0.6, FP4-Blackwell ~0.4) with a FREE draft:
  it beats the chain only if q < 0.01-0.07 (phase-1 < ~1-2 ms, essentially free). FP8,
  FP4-Blackwell, and even free-draft all LOSE.

**One root cause for all three:** on a confident acceptance profile (`h(1)=0.875` already
high), a chain captures most of the accept length, so the tree's advantage over it is tiny
(~+2.5%). There is almost no headroom for pruning/cascading to harvest, and any extra stage
-- however cheap or quantized -- is pure cost. (Quantizing phase-1 lowers its cost but
cannot manufacture headroom the profile doesn't have.) The cascade would only pay on a
high tree-vs-chain-gap profile (high-entropy / ambiguous decoding) -- a falsifiable open
hypothesis, NO-GO here.

**What survives EAGLE:** exactly one thing, and it is draft-agnostic -- the **verify-comm-
scaling** result (Sec. 1/2): on comm-bound MoE use a CHAIN (high batch) or a SMALL tree
(low batch), never a wide tree, and don't add a pruning/cascade stage. This improves
EAGLE-on-MoE too.

## 6. Self-spec vs conventional speculative decoding (EAGLE/MTP)

| axis | conventional (EAGLE/MTP) | self-spec (this work) |
| --- | --- | --- |
| draft | separate trained head/model | SAME model, comm-free local routing + FP4 experts |
| training | per-model training required | **training-free** |
| draft cost / token | cheap (~1-3 layers, ~2-3 ms @ B512) | expensive (full comm-free forward, ~23 ms @ B512) |
| acceptance beta | high (trained to match, ~0.8, 4-5 tok) | lower (local 0.85 / FP4 0.92) |
| extra memory | tiny (the head) | resident expert cache (1-8 GB, Sec. 3) |
| draft communication | comm-free (dense head) | comm-free (local routing) |
| lossless | yes (rejection sampling) | yes (rejection sampling) |
| out-of-the-box on any MoE | no (train a head) | **yes** |

**The uncomfortable truth:** EAGLE already provides a **comm-free draft**, so the project's
original distinctive claim ("remove the all-to-all from the draft") is *not* novel -- EAGLE
has it. And EAGLE's draft is **~8-10x cheaper** than our local-routing draft (a small head
vs a full-model forward), so even in the comm-bound regime EAGLE gets a higher speedup:

```
serving batch (B=512, k=1, comm-bound MoE; EAGLE draft estimated from head-size + measured verify):
  EAGLE:     (1.8)*48 / (3 + 48)  ~= 1.69x
  self-spec: (1.85)*48 / (23 + 48) ~= 1.25x
```

EAGLE also wins on memory (no expert cache) and acceptance (trained). **So conventional
spec dominates self-spec on every axis except one: training-free / out-of-the-box.**

**Where self-spec is still the right tool:** when you will not or cannot train a head --
proprietary or rapidly-rotating models, no training pipeline, or a need to run on *any* MoE
immediately. There, self-spec gives a lossless comm-bound speedup (~1.2-1.35x single-node)
with no extra model to train, store, or serve.

**The convergent ceiling (the real scientific point):** in comm-bound MoE both methods are
capped by the **verify communication**, not the draft -- the all-to-all over `B*k` tokens.
Phase 27/31 show this wall forces chains/small trees for *either* draft. So the draft
method changes how *close* you get to the ceiling (EAGLE's cheaper draft gets closer), but
the ceiling itself is set by the verify comm and is draft-agnostic. That analysis -- the
durable contribution -- improves conventional spec decoding on MoE as much as it does ours.

## 7. Bottom line, regime splits, and ceilings

**The design works and is lossless. On a single PCIe server it is a modest, throughput-
regime win:**

| quantity | low batch (latency) | high batch (serving) |
| --- | --- | --- |
| speedup | ~1.2x (+8% from small trees) | ~1.27-1.35x (k=1) |
| draft memory / device | 1-4 GB (dynamic cache, beta 0.99) | 8 GB (globally-hot 0.5E + skip-cold, beta ~0.91) |
| accept length | trees help (verify latency-bound) | k=1 (verify comm-bound) |

**The central tension:** the comm-bound SPEEDUP wants HIGH batch (f rises with batch); the
memory WIN wants LOW batch (small union). They do not co-occur.

**Fundamental ceilings (all measured):**
- Speedup capped ~1.35x single-node -- the comm-free draft only saves ~half the PCIe step.
- Comm-free draft DEPTH capped -- in-cycle local draft tokens degrade deep predictions
  (even the oracle decays).
- Verify cost scales with verified tokens (k / tree size) -- the comm-bound penalty the
  draft is trying to avoid reappears in the verify.

**Where the real win lives (hardware-gated):** inter-node IB (network hop, far more
comm-bound -> 1.6-2.3x even at low batch). Single-server PCIe is the milder cousin.

---

## 8. Reproducibility

| result | script |
| --- | --- |
| forced-PCIe EP decode + f | `24_commbound_throughput/bench_commbound.py` (NCCL recipe above) |
| real multi-token accept (k-sweep) | `24_commbound_throughput/stageB_ksweep.py` |
| accept-length-per-GB frontier | `26_accept_length_frontier/frontier.py` |
| tree h(b) + optimizer + attention validation | `27_tree_drafting/{tree_hitrate,tree_optimize,tree_verify,control_lossless}.py` |
| dynamic cache / depth decay / skip-cold / union | `28_dynamic_cache/{dynamic_cache,depth_decay,skip_cold,union_coverage}.py` |
| cross-model skew | `29_cross_model/cross_model_skew.py` |
| EAGLE/cascade survival (Phase 31) | `31_eagle_tree_prune/{prune_frontier,prune_cascade_economics}.py` |

**Open / not done:** integrated distributed step-level driver (B2b-full); inter-node IB +
DBO baseline (need a real multi-node box); per-model beta refinements (renorm recovery,
dynamic cache, depth decay measured on Qwen3 only).
