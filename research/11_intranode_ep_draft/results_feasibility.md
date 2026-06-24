# Results: Hierarchical-EP Draft Feasibility

Date: 2026-06-24

Verifies the two prerequisites for the intra-node-EP draft (draft = intra-node EP
/ NVLink only; verify = full EP / inter-node IB): **acceptance high enough** and
**draft step short enough**.

## Question 2 first: acceptance (measurable now)

Intra-node draft acceptance is governed by how many experts the node holds and
which ones. With the **realistic contiguous shard** (`E/nodes` arbitrary experts,
no replication), acceptance is weak -- especially for fine-grained models:

| #nodes | experts/node | Qwen3-30B | GPT-OSS-20B |
| ---: | ---: | ---: | ---: |
| 2 | E/2 | 0.33 | 0.59 |
| 4 | E/4 | 0.12 | 0.46 |
| 8 | E/8 | 0.05 | 0.33 |

But the node aggregates `G` GPUs, so a **per-node high-coverage draft cache** is
cheap: holding ~40-64 well-chosen experts costs only `~40/G = 5-8` experts per GPU
(vs ~4 natively owned at EP32). With that cache, acceptance reaches the Phase
09/10 levels:

| per-node cache (experts) | per-GPU (G=8) | Qwen3 acc | source |
| ---: | ---: | ---: | --- |
| 32 | 4 | 0.47 (static) / 0.74 (warm) | Phase 09 / 10 |
| 48 | 6 | 0.69 | Phase 09 |
| 64 | 8 | 0.78 | Phase 09 |

**Verdict (acceptance): feasible.** Naive partition is too weak for fine-grained
models, but a small per-node cache (a few experts/GPU) lifts 2-4 node acceptance
to ~0.74-0.78 -- in range.

## Question 1: is the draft step short enough?

### Measured

| Quantity | Value |
| --- | --- |
| NVLink all-to-all, decode payloads (per collective) | ~27-40 us (use 30 us) |
| Collectives per decode step | `2 x num_layers` (Qwen3: 96, GPT-OSS: 48) |
| Intra-node A2A per step | Qwen3 ~2.9 ms, GPT-OSS ~1.4 ms |
| Per-token compute (HF eager, single GPU) | Qwen3 94.9 ms, GPT-OSS 39.4 ms |

Caveat: the HF-eager compute is an **overestimate** (naive per-expert loop at
batch 1); optimized serving is far faster (~12 ms / ~6 ms used as the realistic
basis). Inter-node IB latency is **modeled** (50/100/200 us per collective)
because forcing a slow NCCL transport crashes here.

### Speedup envelope (num_spec = 3 draft tokens)

Draft = compute + intra-node A2A; verify = compute + inter-node A2A;
`f_inter` = exposed inter-node fraction of a step.

| compute | IB/coll | f_inter | speedup b=0.5 | b=0.74 | b=0.9 |
| --- | ---: | ---: | ---: | ---: | ---: |
| optimized ~12ms (Qwen3) | 50 us | 0.11 | 0.51 | 0.74 | 0.94 |
| optimized ~12ms | 100 us | 0.31 | 0.61 | 0.88 | 1.12 |
| optimized ~12ms | 200 us | 0.52 | 0.77 | 1.11 | 1.41 |
| HF-eager 95ms | 100 us | 0.06 | 0.49 | 0.71 | 0.90 |

(GPT-OSS gives the same `f_inter`/speedup at matched compute, with smaller
absolute times.)

### Reading

- The draft step is **not dramatically cheaper** than verify. It still pays full
  compute and the intra-node NVLink all-to-all (~3 ms/step Qwen3). Its only saving
  is the inter-node hop, so "short enough" is entirely a question of `f_inter`.
- The method needs **both** a high inter-node fraction (`f_inter >= ~0.5`, which
  requires ~200 us/collective IB latency *and* lean compute) **and** high
  acceptance (`b >= 0.74-0.9`). In that corner it gives ~1.1-1.4x.
- At a modest inter-node fraction (`f_inter ~ 0.3`, IB 100 us) it is marginal:
  0.88 at b=0.74, 1.12 at b=0.9.
- If compute dominates (HF-eager, or a fast interconnect), `f_inter` collapses
  and the method loses outright.

**Verdict (timing): borderline, corner-dependent, and unverifiable here.** The
draft is communication-*reduced*, not communication-free; it wins only when the
inter-node all-to-all is a large fraction of the step. That fraction needs real
multi-node IB to measure -- single-node NVLink is too fast and forced-slow NCCL
crashes (same wall as Phase 02).

## Combined verdict

| Prerequisite | Status |
| --- | --- |
| Acceptance high enough | Feasible: ~0.74 at 2-4 nodes with a small per-node cache |
| Draft step short enough | Borderline: only if `f_inter >= ~0.5` (high IB latency, lean compute); needs multi-node to confirm |

The hierarchical-EP framing fixes the axis that was previously fatal (acceptance:
moved from ~0.1 to ~0.74 at 2-4 nodes) at low memory cost. The remaining risk is
entirely on timing: whether the inter-node all-to-all is a large enough fraction
of the decode step. This is the same `f`-dependent narrow regime Phase 01
identified, now grounded in measured NVLink latency, the `2 x num_layers`
collective count, and realistic acceptance. A go/no-go on timing requires a 2-4
node IB testbed (or DeepEP's hierarchical low-latency profiler).

## Recommendation

Proceed to the full design only if a multi-node profile shows `f_inter >= ~0.4-0.5`
at the target batch size after DeepEP low-latency. Otherwise the acceptance gain
does not convert to speedup. The acceptance side is settled and favorable; the
timing side is the gate, and it is hardware-bound.
