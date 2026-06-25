# Results: Single-Node Injection Emulation of Inter-Node A2A

Date: 2026-06-25. Real Qwen3-30B-A3B decode, attention-DP(4) + EP, dummy weights,
naive AgRs all-to-all, `enforce_eager=False`. Per-collective exposed-A2A latency
injected on the GPU stream (captured into the CUDA graph). See `README.md` for why
this replaces real-IB loopback (which the fabric here refuses).

## 1. Measured: the engine does ~145 exposed collectives per decode step

| B/rank | global | S(0) ms | S(163) ms | S(320) ms | N (coll/step) | f@163 | f@320 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4 | 9.04 | 32.65 | 55.30 | 145 | 0.723 | 0.837 |
| 8 | 32 | 11.79 | 35.34 | 58.02 | 144 | 0.666 | 0.797 |
| 32 | 128 | 14.74 | 39.00 | 62.03 | 148 | 0.622 | 0.762 |
| 64 | 256 | 16.67 | 40.05 | 63.23 | 145 | 0.584 | 0.736 |

`N ~= 145` is flat across batch (it is a structural count: ~3 AgRs collectives --
router-logit gather + dispatch + combine -- per MoE layer x 48 layers). This is the
**real per-engine collective count**, the quantity Phase 15 had to assume.

`f = N*d / S(d)` is the exposed-A2A fraction. At the DeepEP per-collective
latencies (163/320 us) it is **0.58-0.84**, highest at low batch -- exactly the
regime the scheme targets.

## 2. The key result: break-even exposed latency is tiny (~11-18 us/collective)

Speedup of a lockstep `k`-draft + 1-verify cycle (draft collective-free, verify
exact) with one-step acceptance `beta`:

```
speedup(k,f) = (1-beta^(k+1))/(1-beta) / ((k+1) - k*f)
```

At `k=1` this exceeds 1 iff `f > 1-beta`, i.e. exposed per-collective latency

```
d* = (1-beta) * S_base / (N * beta)
```

| beta (source) | break-even f* | d* per collective |
| --- | ---: | ---: |
| 0.85 (affinity G=2, Phase 13) | 0.150 | **11.0 us** |
| 0.78 (EPLB G=4 R~2x, Phase 15) | 0.220 | **17.6 us** |
| 0.95 (FP8 draft, Phase 18) | 0.050 | **3.3 us** |

**Because a step pays ~145 collectives, the per-collective exposed latency needed
to break even is only ~10-18 us** -- far below DeepEP's 163-320 us collective cost.
The method wins unless DBO hides ~>90% of every collective at low batch.

## 3. Speedup vs exposed per-collective latency (low batch, global 4)

| d (us) | f | beta=0.85 | beta=0.78 | beta=0.95 |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 0.138 | 1.00 | 1.00 | 1.05 |
| 25 | 0.286 | 1.08 | 1.04 | 1.18 |
| 50 | 0.444 | 1.22 | 1.14 | 1.40 |
| 100 | 0.615 | 1.48 | 1.35 | 1.82 |
| 163 | 0.723 | 1.76 | 1.56 | 2.30 |
| 250 | 0.800 | 2.08 | 1.80 | 2.87 |
| 320 | 0.837 | 2.29 | 1.95 | 3.29 |

`d` here is the **exposed-after-overlap** latency. d=163/320 = DeepEP's full
collective with zero DBO overlap (the optimistic ceiling: 1.6-3.3x). A conservative
exposed residual of d=50-100 us still gives **1.1-1.8x lossless**.

## 3b. Generalization: GPT-OSS-20B (24 layers, MXFP4) confirms N is structural

Same harness, `openai/gpt-oss-20b` (dummy MXFP4, TRITON Mxfp4 MoE backend):

| global | S(0) ms | S(163) | S(320) | N | f@163 | f@320 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 8.59 | 19.01 | 30.37 | 68 | 0.548 | 0.717 |
| 32 | 8.13 | 18.79 | 30.15 | 69 | 0.567 | 0.730 |
| 128 | 8.11 | 19.57 | 30.93 | 71 | 0.586 | 0.738 |
| 256 | 9.13 | 20.63 | 31.91 | 71 | 0.557 | 0.714 |

| | Qwen3-30B-A3B | GPT-OSS-20B |
| --- | ---: | ---: |
| MoE layers | 48 | 24 |
| quant | bf16 | MXFP4 |
| **N (coll/step)** | **145** | **70** |
| N / layer | 3.02 | 2.92 |
| f@163 (low batch) | 0.72 | 0.55 |
| f@320 (low batch) | 0.84 | 0.72 |
| break-even d* (beta=0.85) | 11 us | 22 us |
| speedup@163 (beta=0.85) | 1.76x | 1.38x |
| speedup@320 (beta=0.85) | 2.29x | 1.74x |

**N is ~3 collectives per MoE layer in both** (independent of architecture and of
bf16-vs-MXFP4) -- it is a structural count, not a per-model accident. The exposed
fraction and speedup are **lower for GPT-OSS because it has half the layers** -> half
the collectives per step -> less exposed A2A relative to compute. The favorable
corollary: **the win scales with MoE depth**, so the real multi-node targets
(DeepSeek-V3 ~61 layers, etc.) sit *above* both test models -- more collectives/step,
higher f, bigger amortization. Even the shallower GPT-OSS keeps break-even at
~22-36 us/collective, still far below DeepEP's 163-320 us.

## 4. Honest caveats (what this does and does not prove)

- **Fully-exposed upper bound.** The naive AgRs path has no DBO, so injecting the
  full `d` assumes zero compute/comm overlap. A real DeepEP+DBO system hides some of
  every collective; the true exposed `d` -- and thus the operating point on the §3
  curve -- needs real multi-node hardware. This phase brackets the curve and shows
  the **break-even bar is low** (~15 us), not where exactly the system lands.
- **One-step `beta` is a proxy.** Real multi-token acceptance is lower; verify must
  be exact bf16 for losslessness (the §1 measurement already uses full-routing
  verify steps).
- **No cross-node switch geometry.** Injection models added per-collective latency,
  not the full inter-node pipeline (stragglers, new-arrival queueing at the verify
  boundary -- Phase 19 threats).
- **Lossless by construction:** every draft step is verified by an exact full-EP
  step; the gain comes only from amortizing communication, never from approximation.

## 5. Bottom line

On the real engine, low-batch MoE decode pays **~145 exposed collectives/step**, so
the inter-node all-to-all is a large fraction of the step (`f`=0.58-0.84 at DeepEP
latencies) and the **break-even exposed latency is only ~10-18 us/collective**. A
collective-free local draft + exact verify is therefore **1.1-3.3x lossless** across
the plausible exposed-latency range -- contingent on the one number this node cannot
produce: how much DBO hides at the target batch. That single gate now has a precise
threshold (`d* ~ 15 us`) to test on a 2-node IB rental.
