# Results: G>=4 Replication (EPLB / Expert-Cache) Speedup Envelope

Date: 2026-06-24

Can EPLB / expert-cache replication make the hierarchical draft pay off at G>=4?
Acceptance(phi) from Phase 09 (M = phi*E mass-optimal experts per node) x the
latency-bound comm model (measured NVLink 30us/coll, modeled IB, n=3 draft
tokens). `R@G4` = replication factor over native EP at G=4 (`phi*4`);
`exp/GPU` = local experts per GPU at 8 GPUs/node.

## Qwen3-30B-A3B (E=128, 48 layers, compute~12ms)

| phi | M | acc | R@G4 | exp/GPU | sp(IB=50) | sp(IB=100) | sp(IB=200) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.25 | 32 | 0.470 | 1.0 | 4 | 0.49 | 0.59 | 0.74 |
| 0.38 | 48 | 0.693 | 1.5 | 6 | 0.69 | 0.82 | 1.03 |
| 0.50 | 64 | 0.781 | 2.0 | 8 | 0.78 | 0.94 | 1.18 |
| 0.75 | 96 | 0.911 | 3.0 | 12 | 0.96 | 1.14 | 1.44 |

Break-even phi: IB=50us never; IB=100us at phi>=0.75 (R=3); **IB=200us at phi>=0.38 (R=1.5, 6 exp/GPU)**.

## GPT-OSS-20B (E=32, 24 layers, compute~6ms)

| phi | M | acc | R@G4 | exp/GPU | sp(IB=50) | sp(IB=100) | sp(IB=200) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.25 | 8 | 0.537 | 1.0 | 1 | 0.54 | 0.65 | 0.81 |
| 0.50 | 16 | 0.625 | 2.0 | 2 | 0.62 | 0.74 | 0.93 |
| 0.75 | 24 | 0.758 | 3.0 | 3 | 0.76 | 0.90 | 1.14 |

Break-even phi: only IB=200us reaches it, at phi>=0.75 (R=3).

## Findings

**1. Replication does make G=4 work -- monotonically.** Because decode all-to-all
is latency-bound, replicating to fraction phi raises acceptance (Phase 09) while
the verify inter-node collective still fires (the draft still removes its full
latency). So speedup rises monotonically with phi: push replication to the memory
budget. This confirms the EPLB/cache idea is sound, not self-defeating, at low
batch.

**2. But the win is gated on inter-node per-collective latency.**
- IB ~50us/coll: no win at any affordable phi.
- IB ~100us/coll: win only at heavy replication (phi>=0.75, R=3x).
- IB ~200us/coll: win at modest replication -- **Qwen3 nets ~1.18x at phi=0.5
  (R=2x, 8 experts/GPU, acceptance 0.78); break-even at R=1.5x.**

**3. Concrete spec for G=4 (Qwen3, the favorable case):** replicate hot/co-activated
experts (via EPLB) to ~half the pool per node (R~=2x, ~8 experts/GPU), giving
acceptance ~0.78; this nets ~1.18x **iff** the exposed inter-node collective
latency is >= ~200us per collective. GPT-OSS is harder (fewer layers -> less
inter-node latency to amortize per step), needing R~=3x even at IB=200us.

## Honest caveats

- **Still inter-node-latency-bound.** The whole envelope hinges on IB per-collective
  latency >= ~100-200us. DeepEP low-latency and DBO overlap *reduce* exposed
  inter-node latency, pushing toward the unfavorable column -- so the method must
  beat that stack, not raw IB. Measuring real exposed IB latency after overlap is
  the open multi-node measurement.
- **Memory.** R = phi*G grows with G (R~2x at G=4, ~4x at G=8 for phi=0.5). EPLB
  defrays part (replication you need for load balancing anyway), but not all.
- **Batch ceiling.** Latency-bound monotonicity holds at low batch; above the
  bandwidth crossover the phi-tradeoff returns.
- One-step acceptance proxy; multi-token would be lower.

## Conclusion

EPLB / expert-cache replication **is** a viable way to reach G=4 acceptance, and
because decode A2A is latency-bound it does not erode the comm advantage at low
batch. The concrete operating point (Qwen3): R~=2x replication (~8 experts/GPU) ->
acceptance ~0.78 -> ~1.18x, **conditional on inter-node collective latency >=
~200us after overlap.** That single number -- exposed inter-node A2A latency on a
real 2-4 node IB setup after DeepEP/DBO -- remains the sole go/no-go gate.
