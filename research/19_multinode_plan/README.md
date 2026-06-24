# Phase 19: Novelty Delineation + Multi-Node Experiment Plan

Source phases: all of `00`-`18`. This is the standing reference for the multi-node
stage -- the one regime where the project's distinct contribution lives.

## Objective

State precisely what is novel about Self-MoE-spec relative to published work, and
specify the multi-node experiment that demonstrates it. Everything provable on a
single node is done; the novel claim is inter-node and needs a 2-4 node IB testbed.

## 1. Novelty delineation vs prior work

| Prior work | Mechanism | Bottleneck / axis | Setting | Overlap with us |
| --- | --- | --- | --- | --- |
| **QuantSpec** (ICML 2025) | quantized-weight + 4-bit KV draft, full-precision verify | KV-cache memory | single-device, long-context, dense | The *quantized-draft mechanism* (= our Phase 18 FP8-draft). Not MoE/EP/comm. |
| Speculative Decoding Meets Quantization (2505.22179) | spec x quant compatibility + hierarchical design | compute/memory | single-device | Same axis as QuantSpec; not communication. |
| SS-MoE | expert-subset self-spec | on-device expert memory/offload | single-device MoE | Our local-routing-for-*memory* (Phase 16). Not multi-node comm. |
| Semantic Parallelism (2503.04398) | affinity placement + token scheduling | all-to-all *volume* (bandwidth) | multi-node MoE | We *use* its placement (Phase 13); it is not speculation. |
| MoE-Spec / EVICT / Cascade | verify-side expert budgeting / tree truncation | verify expert expansion | MoE spec decode | Different problem (restrict verify, not draft). |

**What every one of these leaves untouched:** using a draft to amortize the
**inter-node all-to-all** of multi-node MoE EP -- i.e. converting expert locality
into *fewer collectives per accepted token*. That axis (communication, not
KV/weight memory) and that setting (multi-node MoE EP, low-batch decode) are ours.

## 2. The distinct claim

```text
At low-batch multi-node MoE EP decode, a draft that routes only to device-/node-
local experts (optionally also FP8) is collective-free, so it removes the exposed
inter-node all-to-all that even DeepEP low-latency + DBO cannot hide at low batch.
Verified by exact full-EP routing, this is lossless and nets a TPOT speedup that a
single-device quantized-draft self-spec (QuantSpec-style) cannot get -- because that
draft still pays the all-to-all, ours does not.
```

Two sub-claims, in order of strength:
1. Beats full-EP bf16 decode at low batch.
2. Beats full-EP **quantized-draft self-spec** (QuantSpec-style) at low batch --
   the strong claim, isolating the *communication* contribution from the (published)
   *quantization* one. Strongest of all: **compose** them -- (FP8 + local) draft --
   since intra-node they are redundant but inter-node they attack different axes
   (compute vs communication).

## 3. Inputs already settled on one node (do not re-derive)

| Quantity | Value | Source |
| --- | --- | --- |
| Acceptance, affinity placement, G=2 | ~0.83-0.86 | Phase 13 |
| Acceptance, EPLB replication, G=4 (R~2x) | ~0.78 | Phase 09/15 |
| Acceptance, FP8 draft vs bf16 target | 0.954 | Phase 18 |
| Local routing + FP8 | redundant intra-node (compose only inter-node) | Phase 18 |
| Intra-node EP-width latency effect | <=7% (no win) | Phase 12 |
| Timing break-even (modeled) | net win iff exposed inter-node A2A >= ~200us/coll | Phase 15 |
| DeepEP low-latency A2A (reference) | dispatch ~163us, combine ~320us/coll | DeepEP |
| Draft must be collective-free across the group | lockstep cycles, no token-level fusion | Phase 00 |

## 4. Multi-node experiment design

**Hardware:** 2-4 nodes x 8 GPUs, InfiniBand, DeepEP low-latency installed; DBO
enabled. Attention-DP + EP (DeepSeek-style), prefill disaggregated or reported
separately.

**Model:** a DeepSeek-scale MoE (many layers x many experts -> high per-collective
inter-node latency -> favorable; also brings a shared expert, the one untested
acceptance aid). Smaller fallback: any MoE that exercises real cross-node EP.

**The scheme:** lockstep cycles -- k local-draft steps (intra-node EP only, no
inter-node all-to-all; optionally FP8) then 1 full-EP bf16 verify step; standard
rejection sampling (lossless). Affinity placement (Phase 13) and/or EPLB
replication to R~2x (Phase 15) for acceptance.

**Baselines (must beat, tuned):**
1. full-EP bf16 plain decode;
2. full-EP bf16 + DeepEP low-latency + DBO;
3. full-EP **quantized-draft self-spec** (QuantSpec-style: quantized draft, full
   verify, but draft still does the all-to-all) -- the key comparison that isolates
   our communication contribution.

**Measurements:**
- Exposed inter-node all-to-all latency per collective, after DeepEP/DBO overlap,
  at the target low batch (the gate quantity).
- Real multi-token acceptance `beta` (not the one-step proxy used on one node).
- End-to-end TPOT and throughput vs every baseline; crossover batch where the win
  disappears; replication memory cost.

## 5. Go / No-Go

- **Go** if exposed inter-node A2A >= ~200us/coll after overlap AND the (FP8+local)
  draft + full verify beats baseline 3 (quantized-draft self-spec) on TPOT at low
  batch, losslessly. Phase 15 then predicts ~1.2-1.4x+ at R~2x, beta~0.78-0.95.
- **No-Go / negative result** if exposed A2A is small after DeepEP/DBO (overlap
  already hides it) -- then the communication has nothing left to amortize and the
  method collapses to the published quantization story. Write up the
  `exposed-A2A-after-overlap` characterization as the contribution either way.

## 6. Implementation requirements (vLLM)

- A draft path that masks MoE routing to local (node) experts and runs intra-node
  EP only (skip the inter-node dispatch/combine) -- partially prototyped by the
  research hooks in Phase 02 (`VLLM_SELF_SPEC_*`).
- Lockstep draft/verify cycle scheduler (uniform k across the EP group; new
  arrivals join at cycle boundaries; PD disaggregation on the decode side).
- Two CUDA-graph shapes (collective-free draft; full-EP verify).
- Optional FP8 draft expert weights resident alongside bf16 verify weights
  (~1.5x expert memory; runtime downcast saves nothing -- Phase 18).
- EPLB / affinity placement integration for acceptance.

## 7. Threats to validity

- DeepEP low-latency + DBO may hide most inter-node A2A at the target batch ->
  exposed `f` too small -> no room (the central risk; measure first).
- Lockstep cycle overhead, cross-rank stragglers at the verify boundary, and
  new-arrival queueing can erode the gain -- include them in TPOT.
- Multi-token `beta` will be below the one-step proxy; verify must use exact bf16.
- Replication memory (R~2x at G=4) must fit alongside KV cache at the target batch.

## Expected artifact

A multi-node TPOT comparison (this plan, executed) showing whether
(FP8+local)-draft + full-EP verify beats tuned DeepEP+DBO and a QuantSpec-style
quantized-draft self-spec at low-batch multi-node decode -- the project's
go/no-go, and the result that is novel relative to QuantSpec.
