# Results: Exact EP-Width Decode Latency

Date: 2026-06-24

Real vLLM decode latency (ms/token) for `Qwen/Qwen3-30B-A3B` (dummy weights),
single-node NVLink, GPUs 2-5. EP4 = `TP4+EP` (4 GPUs); EP2 = `TP2+EP` (2 GPUs).

## Equal per-engine batch (pure layout effect)

| batch | EP4 (4 GPU) | EP2 (2 GPU) | EP2 / EP4 |
| ---: | ---: | ---: | ---: |
| 1 | 4.64 | 5.01 | 1.08x |
| 2 | 5.35 | 5.80 | 1.08x |
| 4 | 6.03 | 7.11 | 1.18x |
| 8 | 7.16 | 9.05 | 1.26x |
| 16 | 8.53 | 11.46 | 1.34x |
| 32 | 9.87 | 13.88 | 1.41x |

At equal batch, **EP4 is faster than EP2** and the gap grows with batch. More
GPUs = more compute and fewer experts/weights per GPU; the larger all-to-all over
NVLink does not overcome that. So on NVLink, a *smaller* EP does not make the step
cheaper -- it makes it more expensive.

## Throughput-matched: EP4xDP1 (batch B) vs EP2xDP2 (B/2 per engine)

This is the apples-to-apples 4-GPU comparison: same hardware, same global batch.

| global B | EP4xDP1 = L4(B) | EP2xDP2 = L2(B/2) | winner | gap |
| ---: | ---: | ---: | --- | ---: |
| 2 | 5.35 | 5.01 | EP2xDP2 | 6.7% |
| 4 | 6.03 | 5.80 | EP2xDP2 | 3.9% |
| 8 | 7.16 | 7.11 | EP2xDP2 | 0.7% |
| 16 | 8.53 | 9.05 | EP4xDP1 | 6.0% |
| 32 | 9.87 | 11.46 | EP4xDP1 | 16.1% |

EP2xDP2 (lower EP, higher DP) is only marginally faster at **low batch** (<=4,
by <=7%), ties at B=8, and loses at B>=16. The EP2xDP2 numbers are even optimistic
-- they are a single EP2 engine; two concurrent engines would contend on the
shared NVSwitch.

## Interpretation

- On NVLink, the EP-width choice barely moves decode latency (within +-6-16%),
  with a smaller EP helping only marginally at very low batch.
- For the hierarchical draft, the draft uses a smaller (intra-node) EP. This
  measurement says that on NVLink that buys **at most ~3-7% at low decode batch**
  -- far too little to pay for speculation, which runs several draft steps per
  verify.
- Crucially: with **both** layouts intra-node (no inter-node hop), reducing EP
  does not speed up the step at all -- it slows it, because you give up GPUs and
  compute. Therefore the entire benefit of the hierarchical draft hinges on the
  **inter-node** all-to-all penalty, which this single node cannot produce.

## Verdict

This is the real-measurement version of the Phase 11 timing gate. It confirms,
with measured vLLM latencies rather than a model, that:

```text
Intra-node, EP-width is a small latency lever (<=~7% at low batch, negative at
high batch). The hierarchical-EP draft can only pay off when the all-to-all it
removes is inter-node (IB), not intra-node (NVLink).
```

So the timing go/no-go is now firmly established as **inter-node-bound**: it
cannot be settled on this node, and requires a 2-4 node IB testbed (or DeepEP's
hierarchical low-latency profiler) to measure the inter-node fraction directly.
The acceptance side (Phase 11) is favorable; the timing side remains gated on
hardware we do not have.
