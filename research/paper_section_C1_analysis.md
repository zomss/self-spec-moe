# Section 3: Speculative Decoding is Communication-Bound on MoE-EP

*(Full-prose draft of contribution C1. Numbers are measured on 8xH100 (NVSwitch) with the
EP all-to-all forced over PCIe; see §3.2 for the methodology. Figure/table numbers are
placeholders.)*

## 3.1 Expert-parallel decode and its communication

A Mixture-of-Experts layer routes each token to a small number `k_top` of its `E` experts.
At serving scale the experts do not fit on one device, so they are sharded across devices
with **expert parallelism (EP)**: each device holds a slice of the experts, and every MoE
layer performs two all-to-all collectives per step -- a *dispatch* that sends each token's
hidden state to the device(s) holding its chosen experts, and a *combine* that returns the
expert outputs. A decode step therefore issues on the order of two-to-three collectives per
MoE layer; for the 48-layer model we study this is roughly **145 exposed collectives per
step**. On a single token, each collective moves `B * d * k_top` elements, where `B` is the
number of concurrent sequences and `d` the hidden size.

The cost of this communication relative to the layer's arithmetic is governed by the
machine balance -- the ratio of the device's compute throughput to the interconnect
bandwidth. On NVLink/NVSwitch (measured peer bandwidth ~389 GB/s) the all-to-all is cheap
relative to the FFN arithmetic, and decode is compute-bound. Off NVLink it is not: we
measure PCIe peer-to-peer at ~55 GB/s and host-staged PCIe at ~27.5 GB/s -- 7x to 14x
slower than NVLink -- and inter-node networks add a further hop. In these settings the
expert all-to-all, not the FFN, sets the decode latency. This is the **communication-bound
regime**, and it is the regime in which most MoE serving outside a single NVLink domain
operates: multi-node deployments, and single nodes whose GPUs are not fully NVLink-connected.

## 3.2 Measuring the communication-bound regime

**Forcing PCIe on an NVLink box.** Our hardware is NVLink-connected, so to observe the
communication-bound regime directly we must route the EP all-to-all over PCIe. This is more
subtle than it appears: setting `NCCL_P2P_DISABLE=1` alone does *not* take traffic off
NVLink, because NVLS (NVLink multicast over the NVSwitch) is a separate transport that
remains active. Disabling all three of point-to-point P2P, NVLS, and InfiniBand
(`NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=1->0 NCCL_IB_DISABLE=1`) forces NCCL onto its
host-staged shared-memory transport -- the channels report `via SHM/direct`, i.e.
GPU->host->GPU over PCIe -- with no NVLink, no hang, and no TCP. Because the framework's EP
collectives are plain NCCL calls (`all_gatherv` / `reduce_scatterv` via pynccl), this
single environment change steers the entire serving path onto real PCIe. We use this as a
faithful, hang-free single-node proxy for off-NVLink MoE-EP serving.

**The measured fraction.** Table 1 reports the decode step time for an attention-data-
parallel + EP configuration of a 30B-parameter, 3B-active MoE, on NVLink versus forced PCIe,
as the global batch grows. On NVLink the step is essentially flat at ~15-17 ms (the
communication is hidden); on PCIe it rises from 23 ms to 48 ms. The implied communication
fraction `f = (T_PCIe - T_NVLink) / T_PCIe` grows from **0.37 at batch 8 to 0.65 at batch
512**, and saturates near **0.62** at higher batch (Fig. 1). Two facts matter for the rest
of the paper: communication is already the *majority* of the step at serving batch, and `f`
*rises with batch* until both communication and compute scale together.

## 3.3 The verify cost model: communication scales with verified tokens

Speculative decoding proposes `n` tokens per cycle (a chain of length `n`, or a tree of `n`
nodes) and **verifies them in a single forward pass** of the target model. The verifier
therefore processes `n` tokens per sequence at once. On MoE-EP this is the crux: a forward
over `n` tokens for `B` sequences routes `B * n` tokens through every all-to-all. The verify
communication is thus

```
    T_comm(n, B)  =  N_coll * L            (fixed: per-collective latency, ~145 collectives)
                   +  (n * B * d * k_top * bytes) / W      (scales with verified tokens)
```

The first term is paid once per verify and is **amortized** over the `n` tokens; the second
**scales linearly with the number of verified tokens** `n * B`. Fitting the measured forced-
PCIe verify step to the total token count `T = B * n` gives

```
    T_verify(T)  =  22.7  +  0.0495 * T     (ms),
```

while the communication-free forward (the same model with the all-to-all elided) is

```
    T_compute(T) =  15.3  +  0.0152 * T     (ms).
```

The intercept of `T_verify` (~23 ms) is the fixed latency floor `N_coll * L`; its slope
(~0.05 ms/token) is the per-token bandwidth term. Subtracting, the all-to-all itself is
~7 ms fixed plus ~0.034 ms per routed token. This decomposition has three immediate
consequences that drive the design:

1. **A single batched verify already amortizes the latency.** Verifying `n` tokens in one
   forward pays `N_coll * L` once, not `n` times -- there is no further "batch the verify"
   gain to capture; speculative decoding already does this.

2. **The verify communication grows with `n` (chain depth / tree width) and with `B`.**
   At low batch the fixed latency dominates, so deeper drafts amortize it and help; at high
   batch the bandwidth term dominates, so every extra verified token costs communication.
   This is why, in the communication-bound regime, the optimal proposal is *small*: at
   serving batch the verify all-to-all of a `k`-token structure is ~`k`x that of a single
   token, so a one-token chain is the throughput optimum, and a chain (minimal breadth) is
   the communication floor for a given accepted length.

3. **The draft should avoid the all-to-all entirely.** Because the verify already pays the
   communication for every committed token, the *draft's* only job is to propose; any
   communication it performs is pure overhead. A draft that routes locally (no all-to-all),
   or a dense draft head, removes it.

We note that (2) and (3) are *draft-agnostic*: they constrain the verify and the draft's
communication regardless of how the draft is produced, and therefore apply equally to a
trained draft head (EAGLE/MTP) and to a self-speculative draft.

## 3.4 Implications

Sections 4-5 follow directly. **(C2 / World B)** Because the verify communication scales
with the number of verified tokens, the wide draft trees used by EAGLE/MTP -- which route
many candidate tokens to accept only the best path -- inflate the verify all-to-all on
MoE-EP; a communication-aware small or confidence-pruned tree recovers most of the accepted
length at a fraction of the communication. **(C3 / World A)** Because the draft need only be
communication-free, and need not be a trained model, a local-routing pass over a small
resident expert cache yields a training-free, lossless self-speculative draft; the
challenge it raises -- keeping the resident experts small while routing locally -- is a
memory problem we address in §5. Both designs, and the negative results that bound them
(§6), are consequences of the single measurement in this section: on MoE-EP off NVLink,
the verify's communication is the cost, and it scales with the tokens you ask it to verify.
