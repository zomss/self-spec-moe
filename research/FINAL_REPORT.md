# Self-MoE-Spec: Final Research Report

Date: 2026-06-24
Scope: Phases 00-15. Status log: [`status.md`](status.md).

---

## 1. The idea

**Self-MoE-spec** is a lossless self-speculative decoding scheme for expert-parallel
(EP) MoE serving:

```text
Draft k tokens using the SAME model, but route MoE layers only to device-local
experts (no expert-parallel all-to-all).
Verify the k drafts with normal full-routing MoE (exact, with all-to-all).
Use standard speculative rejection sampling -> output is lossless regardless of
how badly local routing perturbs the draft.
```

The defining economic fact (Phase 00): unlike EAGLE/MTP/layer-skip, **the draft is
not compute-cheap** -- it still runs full attention + full top-k expert FLOPs
(`c ~= 1`); it only removes the all-to-all. So Self-MoE-spec is purely a
**communication-amortization** scheme: pay the dispatch/combine collective once per
verify cycle instead of once per token.

Consequences fixed at the outset:
- Speedup ceiling at perfect acceptance is `1/(1-f)`, where `f` = exposed all-to-all
  fraction of a decode step. The whole method lives or dies on `f`.
- Token-level draft/verify fusion destroys the gain (reinstates the collective every
  step), so cycles must be **lockstep** and collective-free during draft.
- Target regime: **low-batch, multi-node EP decode**, where the all-to-all is on the
  critical path and FLOPs are idle.

## 2. Two questions, two axes

Every phase ultimately measured one of two things:
- **Acceptance** (`beta`): does a local-only draft survive exact verification often
  enough? Needs `beta >= 0.8` (high `f`) to `>= 0.98` (on top of DBO).
- **Timing** (`f`): is there enough exposed communication to amortize?

The arc is the story of these two axes turning out to be **anti-correlated**, and of
slowly separating *where* and *how* each can be won.

## 3. Phase-by-phase findings

| Phase | Question | Finding |
| --- | --- | --- |
| 00 proposal | Is it sound? | Plausible but narrow; draft is comm-amortization, not compute-cheap; needs lockstep + multi-node. |
| 01 comm envelope | How much room? | `f=0.15` -> ceiling ~1.05-1.14x; `f=0.6` needs `beta>=0.8`; on DBO needs `beta>=0.98`. |
| 02 timing (measured) | Real single-node timing? | Single-node EP weak; forced IB unstable (`IBV_WC_RETRY_EXC_ERR`); built a research delay hook. |
| 03 local routing | Real acceptance proxy? | PowerMoE-3B / Qwen1.5-MoE: `gamma~0.6` at EP2, next-token overlap ~0.37. Negative. |
| 04 placement | Can placement fix it? | Optimized + hot-replicated placement: `gamma~0.68` at best. Top-1 locality high, full top-k low. |
| 05 cheaper drafts | top-1/2 local? | Best ~0.40 sampled acceptance. Insufficient. |
| 06 policy gating | Draft only good tokens? | Simple gating can't isolate a high-acceptance subset (best ~0.40, tiny fraction). |
| 07 recent models | Model artifact? | Qwen3-30B / GPT-OSS-20B: best at EP2 (0.50/0.70), collapse at EP8 (0.12/0.52). **Anti-correlation.** |
| 08 negative writeup | State the result | Scoped negative result for no-shared-expert local-only drafting. |
| 09 rebalancing ceiling | How much replication? | `beta>=0.8` needs `M* ~= 0.5-0.8 E` per device; replication factor `R ~= 0.5N-0.8N` (grows with EP). |
| 10 verify-warmed cache | Exploit verify routing? | LRU/per-request cache halves the needed cache (`R ~= 0.3N-0.5N`); mostly per-request specialization, not recency. |
| 11 hierarchical draft | Flatten EP for draft? | Draft = intra-node EP, verify = full EP. Acceptance feasible; timing needs `f_inter >= ~0.5`. |
| 12 EP-width latency | Real intra-node cost? | vLLM EP2 vs EP4: EP-width changes <=7% at low batch (negative at scale). **Benefit is inter-node-bound.** |
| 13 affinity + gating | Co-schedule like Semantic Parallelism? | **First positive:** G=2 acceptance clears 0.8 (0.86 / 0.84). But G=4 collapses (0.45-0.59). |
| 14 intra-node pattern | Locality pattern intra-node? | Semantic-Parallelism-like locality visible at G=2, but no end-to-end intra-node speedup. |
| 15 phi replication | EPLB/cache at G>=4? | Replication makes G=4 viable (latency-bound, monotonic in `phi`); ~1.18x at `R=2x` iff inter-node A2A `>= ~200us/coll`. |
| 16 intra-node throughput | A throughput axis intra-node? | A local draft activates fewer experts -> ~0.63x MoE-FFN read at serving batch (memory-bound). Real vLLM pins `phi_moe ~= 0.7`; throughput gain ~1.07x (b=0.85) to ~1.12x (b=0.9) at k=2. |

## 4. The two structural results

### 4a. Acceptance and communication advantage are anti-correlated (Phases 07, 09-12)

Increasing EP shards routed experts across more devices, so each device holds fewer
(`E/N`), local coverage `gamma` falls, and acceptance falls. But higher (cross-node)
EP is exactly what makes the all-to-all expensive (raises `f`). **The two
preconditions move in opposite directions along the EP axis:** acceptance is best
where communication is cheap (low EP / intra-node), and collapses where
communication is expensive (high EP / inter-node).

Replication (Phases 09-10, 15) softens this: to hold local fraction `phi` you pay a
replication factor `R = phi*N` (per GPU, amortized across a node's GPUs). But `R`
grows with EP, so the fix gets more expensive exactly where it's most needed.

### 4b. Volume vs latency: Self-MoE-spec and Semantic Parallelism attack different costs

| | Reduces | Helps when | Intra-node win? |
| --- | --- | --- | --- |
| Semantic Parallelism (affinity placement) | all-to-all **volume** | **bandwidth-bound** (prefill, high token count) | **Yes** (its 24.9% TP result) |
| Self-MoE-spec draft | all-to-all **collective latency** | **latency-bound** (low-batch decode) | **No** (NVLink ~30us << threshold) |

This resolves the apparent paradox of Phase 12 (intra-node EP-width barely matters)
vs Semantic Parallelism's intra-node gains: SP wins by cutting *volume* in a
high-volume regime; the self-spec draft removes *collective latency*, which is tiny
intra-node and only large inter-node. They are complementary, on orthogonal axes.

## 5. What is solved, and what is not

**Acceptance: solved (within scope).**
- **G=2:** Semantic-Parallelism-style affinity placement + request scheduling reaches
  `beta ~= 0.84-0.86` -- the first time the project cleared 0.8 (Phase 13).
- **G>=4:** EPLB / expert-cache replication to `phi ~= 0.5` (`R ~= 2x`, ~8 experts/GPU)
  reaches `beta ~= 0.78`, and because decode A2A is latency-bound, replication raises
  acceptance without eroding the comm advantage (Phase 15).
- Scope caveat: tested on **no-shared-expert** models (Qwen3-30B, GPT-OSS-20B); a
  shared expert (DeepSeek/Llama-4/GLM) is an additional, untested acceptance aid.

**Intra-node timing: structurally no win.**
- NVLink all-to-all is ~30 us/collective; EP-width changes decode latency <=7% at low
  batch (Phase 12). The draft removes communication that is already a few percent of
  the step while leaving the dominant cost (compute/memory) untouched and adding a
  draft tax. Net <= 0. No placement/replication/acceptance work changes this.

**Inter-node timing: genuine headroom (the open gate).**
- DeepEP low-latency decode kernels (H800 + 400 Gb/s IB): dispatch ~163-194 us,
  combine ~320-360 us per collective; only ~43-47 GB/s RDMA. A decode step has
  `2*num_layers` collectives -> ~28 ms of all-to-all at DeepSeek scale, matching
  Semantic Parallelism's "up to 59% of forward latency."
- This ~160-360 us/collective is **at or above** the Phase 15 break-even (~200 us at
  `R~=2x`). And the latency that DeepEP/DBO would hide via overlap **cannot be hidden
  at low batch** (no second-microbatch compute to overlap behind) -- which is exactly
  the method's regime.

## 5b. A second axis: intra-node throughput (real but modest)

The whole arc above is about *latency*. Phase 16 opened a different axis. At
moderate-high (memory-bound) batch, MoE decode is bound on **expert weight reads**;
a local-only draft activates fewer unique experts, so it reads fewer weights -- a
**throughput** win that needs no communication and works intra-node, via a
different mechanism (memory, not collective latency).

- Real vLLM single-GPU decode (Qwen3-30B, `num_experts` overridden) pins
  **`phi_moe ~= 0.65-0.76`** (the MoE FFN is ~70% of the decode step) and
  **`T_draft/T_full ~= 0.63`** end-to-end for a `phi=0.5` draft.
- Composed with acceptance (Phase 13/15) and the measured verify creep, the
  throughput gain is **~1.07x (beta=0.85) to ~1.12x (beta=0.9) at k=2**, eroding to
  break-even by k=4 (verify pays for `(k+1)x` tokens).

This is a real, communication-free, fully-intra-node positive -- but small, and
mechanistically overlapping SS-MoE (expert-subset self-speculation). Its cap (~1.1x)
comes from the draft being the *full* model (only ~0.63x cheaper). See Section 9
for what it would take to push throughput much higher.

## 6. The single remaining experiment

Everything resolvable on one node is resolved. The one number not producible here:

```text
Exposed inter-node all-to-all latency per collective, at the target (low) batch,
AFTER DeepEP low-latency + DBO overlap, on a real 2-4 node IB cluster.
```

Go/no-go: if exposed `>= ~200 us/collective`, then at G=4 with `R ~= 2x` EPLB
replication and `beta ~= 0.78`, the model predicts **~1.2-1.4x** TPOT speedup over
the full-EP baseline in low-batch latency-critical decode.

**Recommended test configuration:**
- Hardware: 2-4 nodes x 8 GPUs, InfiniBand, DeepEP low-latency installed.
- Model: a DeepSeek-scale MoE (many layers x many experts -> high per-collective
  latency -> favorable; and tests the shared-expert acceptance aid for free).
- Scheme: intra-node EP draft + full-EP verify, lockstep cycles; affinity placement;
  EPLB replication to `phi ~= 0.5`.
- Measurements: exposed inter-node collective latency after overlap; actual
  multi-token acceptance `beta` (not the one-step proxy used throughout); end-to-end
  TPOT vs the tuned DeepEP(+DBO) baseline.

## 7. Contributions regardless of the multi-node outcome

1. **Acceptance-vs-EP characterization** for local-only MoE drafting across six
   checkpoints (old/recent, fine/coarse, shared/none).
2. **The anti-correlation** between local-draft acceptance and exposed communication,
   and its continuous form (the `phi` trade-off).
3. **The rebalancing ceiling** (`R ~= 0.5N-0.8N` static; `~0.3N-0.5N` verify-warmed)
   -- how much replication useful acceptance costs.
4. **The hierarchical reframing**: draft = intra-node EP, verify = full EP, which
   moves acceptance from fatal to feasible (G=2 via affinity, G>=4 via EPLB).
5. **The volume-vs-latency distinction** that cleanly separates Self-MoE-spec from
   Semantic Parallelism and explains why intra-node is hopeless and inter-node is not.
6. A measured single-node toolchain (acceptance proxies, EP-width latency, all-to-all
   microbench, `phi`-envelope) that fully specifies the multi-node go/no-go.

## 9. Pushing throughput much higher

The intra-node throughput gain is capped at ~1.1-1.2x for one reason: **the draft is
the full model** (full attention + full top-k FLOPs over `M=E/2` experts), so it is
only ~0.63x a step, and with `phi_moe ~= 0.7` halving the MoE read saves only ~35%.
The verify also pays for `(k+1)x` tokens. To go much higher, three families:

**A. Make the draft genuinely cheap (most headroom within speculation) -- but
layer-skip fails (Phase 17, tested).** If the draft were ~0.2-0.3x a step instead of
0.63x, long drafts (k=4-8) would pay off -> potentially ~1.4-2x. The obvious
training-free lever, **layer-skip + local routing**, was built and measured: it does
NOT work. Acceptance drops *super-linearly* with skipped layers (1/8 of layers ->
acceptance 0.83 -> 0.68; half -> 0.09) while cost drops only linearly, so every
amount of skip makes throughput *worse* than pure local routing (best stays ~1.0x).
The draft is the same model and every layer matters, so it cannot be cheaply
approximated. (Caveat: naive contiguous middle-band skip; importance-/learned skip
might trade better, but the slope is steep.) Remaining cheap-draft ideas (top-1 /
shared-expert / adaptive-k) are weaker still on acceptance.

**B. Trained draft head (biggest single lever; leaves training-free).** An
EAGLE/MTP-style ~1-layer head is ~3-5% of a step -> ~2-3x, the standard big win. The
MoE-specific novelty would be a head that is itself local-routing/MoE-aware.

**C. Quantize -- but the lossless use is an FP8 *draft* (Phase 18, measured).**
Decode is memory-bound on expert weights (`phi_moe ~= 0.7`). Serving the whole model
in FP8 gives **~2.2-3.4x** but is **lossy** (a different, lower-quality model). The
*lossless* use respects the constraint that verify must be full precision: an **FP8
draft + bf16 verify**. Measured: FP8 is a near-perfect draft of the bf16 model
(acceptance **0.954** vs the bf16 target) at **0.29x** a bf16 step, so long drafts pay
off -> **~1.9x LOSSLESS throughput** (k=6). This is the cheap-and-accurate draft that
layer-skip never was (FP8 error is tiny where 1/8 layer-skip already fell to 0.73).
Note: under FP8 the expert weights are already small, so **local routing adds no
speedup and only costs acceptance -- FP8 replaces it, not composes with it**. Cost:
~1.5x expert memory (bf16 for verify + FP8 for draft; runtime downcast saves nothing).

**The composition.** Quantized experts x a cheap draft (layer-skip or a small
MoE-aware trained head) for long drafts x EPLB/affinity acceptance x adaptive k --
the levers multiply. Honest framing: if raw throughput is the goal, **quantization
and a cheap (trained) draft dominate**; the local-draft self-spec is a composable
~1.1x rider whose genuine novelty was the *communication* (inter-node latency) angle,
not throughput.

## 8. One-paragraph conclusion

Local-expert self-speculation converts MoE expert locality into fewer all-to-all
collectives per accepted token. On a single NVLink node it cannot win: the
communication it removes is already cheap (a few percent of the step), so the
ceiling is structurally ~1.0x. The acceptance problem that blocked the naive idea is
now solved -- affinity placement reaches `beta >= 0.8` at 2-way grouping, and EPLB
replication extends it to `R ~= 2x` at higher grouping -- and because decode
all-to-all is latency-bound, that replication does not erase the advantage. The
entire remaining question is a single inter-node measurement: whether the exposed
all-to-all latency after DeepEP/DBO at low batch is `>= ~200 us/collective`. Public
DeepEP numbers say the raw budget (~160-360 us/collective, ~28 ms/step at scale) is
well above that and -- crucially -- cannot be hidden by overlap at low batch, which
is precisely this method's regime. The honest verdict: **intra-node, no; inter-node
low-batch decode, plausibly yes, pending one well-defined measurement.**
