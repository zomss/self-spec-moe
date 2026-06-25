# Results: Union-over-k Reuse and Cache-Hit Curves for PCIe-Streamed Experts

Date: 2026-06-25. Real-weight decode traces (HF `transformers`, bf16; GPT-OSS
dequantized from MXFP4), 16 prompts/model (4 buckets x 4), 160 generated tokens
each, per-layer per-position true top-k captured from `output_router_logits`. Raw
routing saved in `data/*_routing.npz`; all tables regenerate with
`python routing_locality.py --tag <t> --reuse`. **Real weights are mandatory** --
dummy weights route uniformly and erase the skew/locality this measures.

This closes the one empirical gap from the PCIe-viability design (Phase 20 +
design discussion): does the streamed-expert working set, after temporal reuse and
a resident cache, fall below the PCIe hideable budget `k * T_step * BW`?

## 1. Union-over-k reuse: the working set grows sublinearly in k

Distinct experts per layer that a `k`-token draft cycle touches (streamed **once**,
reused for all `k` tokens), over `B` concurrent sequences. `reuse = B*k*top_k / distinct`.

### Qwen3-30B-A3B (E=128, top_k=8, 48 layers)

| B\k | k=1 | k=2 | k=4 | k=8 | k=16 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| **1** | 8.0 | 12.7 | **19.6** | 29.2 | 41.1 |
| **2** | 15.1 | 23.1 | 33.8 | 47.1 | 60.8 |
| **4** | 27.0 | 39.2 | 53.1 | 67.5 | 80.7 |
| **8** | 44.5 | 60.0 | 74.3 | 88.1 | 99.2 |

reuse factor (B=1): k=4 → 1.64, k=8 → 2.19, k=16 → 3.11. At B=8,k=16 → 10.3.

### GPT-OSS-20B (E=32, top_k=4, 24 layers)

| B\k | k=1 | k=2 | k=4 | k=8 | k=16 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| **1** | 4.0 | 6.3 | **9.4** | 13.2 | 17.1 |
| **2** | 7.3 | 11.1 | 15.4 | 19.9 | 23.5 |
| **8** | 18.8 | 23.7 | 27.3 | 29.7 | 31.0 |

**Reading.** At the draft's native low batch (B=1), a `k=4` cycle touches only
**~20 of 128** experts/layer (Qwen3, 15% of the pool) / **~9 of 32** (GPT-OSS).
The amortized per-step demand is **4.9 experts/layer** (Qwen3) -- right at the bare
PCIe budget (~4/layer/step, below), so reuse alone gets low batch to the edge of
viable; the cache (§2) tips it over. Reuse strengthens with both `k` and `B`, so it
is exactly the longer cycles that high acceptance wants (Phase 20) that also stream
cheapest.

## 2. Cache-hit curve: strong routing skew, verify-warming helps

Coverage = fraction of a decode position's true top-k already resident. Cache types:
**verify-warmed LRU** (warmed by the previous step's verified routing -- what the
streaming draft actually uses), per-request static (top-C by request count), global
static (top-C by global frequency), random (= C/E).

### Qwen3-30B-A3B

| C | C/E | LRU | per-req static | global static | random |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 0.125 | 0.582 | 0.581 | 0.366 | 0.125 |
| 24 | 0.188 | 0.695 | 0.714 | 0.495 | 0.188 |
| 32 | 0.250 | **0.783** | 0.810 | 0.604 | 0.250 |
| 48 | 0.375 | 0.893 | 0.924 | 0.762 | 0.375 |
| 64 | 0.500 | 0.944 | 0.975 | 0.864 | 0.500 |

### GPT-OSS-20B

| C | C/E | LRU | per-req static | global static | random |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 0.25 | 0.639 | 0.647 | 0.480 | 0.25 |
| 12 | 0.375 | 0.782 | 0.797 | 0.635 | 0.375 |
| 16 | 0.50 | **0.877** | 0.893 | 0.755 | 0.50 |
| 24 | 0.75 | 0.965 | 0.983 | 0.919 | 0.75 |

**Reading.** Routing is strongly skewed: Qwen3 top-25% of experts carry **60%** of
traffic (top-50% → 86%); GPT-OSS top-25% → 48% (top-50% → 75%). A verify-warmed
cache of **25% of experts** covers **78%** (Qwen3) of routing for free; **50%**
covers 94%. LRU ~ per-request static (recency adds little beyond request
specialization, consistent with Phase 10) and both crush the random/global baselines.

## 3. PCIe synthesis: streamed bytes/cycle vs hideable budget

`streamed_bytes = union(k,B) * (1 - LRU_hit(C)) * num_layers * expert_bytes`;
`hideable = k * T_step * BW`. Expert sizes from config at 4-bit: Qwen3 **2.36 MB**,
GPT-OSS **12.4 MB** (3 x hidden x moe_inter). `BW = 50 GB/s` (PCIe Gen5),
`T_step = 9 ms` (Phase 20 low-batch step). `fit = streamed / hideable`; **hidden
iff fit <= 1**.

### Qwen3-30B-A3B -- fit ratio (hidden = bold)

| cache C | B=1, k=4 | B=8, k=1 | B=8, k=4 |
| ---: | ---: | ---: | ---: |
| 0 | 1.23 | 11.2 | 4.67 |
| 16 (12.5%) | **0.52** | 4.68 | 1.96 |
| 32 (25%) | **0.27** | 2.43 | 1.02 |
| 48 (37.5%) | **0.13** | 1.20 | **0.50** |
| 64 (50%) | **0.07** | **0.63** | **0.26** |

### GPT-OSS-20B -- fit ratio

| cache C | B=1, k=4 | B=8, k=1 | B=8, k=4 |
| ---: | ---: | ---: | ---: |
| 0 | 1.56 | 12.5 | 4.52 |
| 8 (25%) | **0.56** | 4.51 | 1.63 |
| 16 (50%) | **0.19** | 1.54 | **0.56** |
| 24 (75%) | **0.06** | **0.44** | **0.16** |

**Verdict: PCIe-viable at low batch with a modest cache.** At B=1, k>=4 the streamed
working set is **fully hidden** behind a 50 GB/s link with only a **12.5-25% cache**
(Qwen3 C=16: fit 0.52; C=32: 0.27). Without any cache, low-batch k=4 is just over
the line (fit 1.23) -- reuse alone nearly suffices and the small cache settles it.
Mid/high batch (B=8) needs a larger cache (Qwen3 C=48 hides k=4 at fit 0.50; C=64 at
0.26) or the skip-cold valve. The pathological B=8,k=1 (no reuse) needs ~50% cache.

## 4. HBM accounting (kept near the bf16 verify shard)

The resident cache is the only persistent extra HBM; the streamed experts use a
small **layer-pipelined buffer** (hold ~2-3 layers in flight, not all 48):

| config (Qwen3) | cache HBM | pipeline buf | total extra | vs FP4 replicate |
| --- | ---: | ---: | ---: | ---: |
| C=16 (low batch) | 1.8 GB | ~0.1 GB | **~1.9 GB** | 14.5 GB |
| C=32 | 3.6 GB | ~0.1 GB | ~3.7 GB | 14.5 GB |
| C=48 (high batch) | 5.4 GB | ~0.1 GB | ~5.5 GB | 14.5 GB |

So the offload design holds GPU memory to **bf16 verify shard + ~2-5 GB**, vs +14.5
GB to replicate the full FP4 set -- the HBM saving the design was for, preserved.

## 5. Mapping to acceptance / speedup

When `fit <= 1` the draft streams its **entire** per-cycle union, so every token's
selected expert is present (quantized) -- draft routing fidelity -> ~1.0, leaving
only quantization error: **beta ~ 0.95** (Phase 18 FP8; somewhat lower at 4-bit).
When `fit > 1` the budget caps streaming; the overflow is **skip-cold**ed to the
resident cache and corrected by verify, so coverage = `cache_hit + streamable
fraction` and beta degrades **gracefully** toward the device-local floor (~0.78,
Phases 13/15). Plugged into the Phase 20 model (`speedup = E[acc](k,beta) /
((k+1) - k*f)`), low-batch B=1 lands near the high-beta branch -> the upper end of
Phase 20's lossless 1.1-3.3x, now without paying the inter-node A2A and without the
full-replicate HBM.

## 6. Caveats

- **Hideable budget scales with draft compute.** `T_step=9 ms` is the Phase 20
  low-batch step; a leaner FP4 draft could be ~0.6x -> budget ~0.6x -> fit ~1.6x.
  Low batch still hides (Qwen3 C=16, k=4: 0.52 -> ~0.83), but the margin is real.
- **Hit-rate applied uniformly.** §3 uses the mean LRU hit across the union; the
  hottest union experts are likeliest cached, so true streamed bytes are a touch
  lower -- the table is mildly conservative.
- **One-step routing.** Union/coverage are exact per position; multi-token acceptance
  compounding is handled by the Phase 20 `E[acc]` term, not here.
- **PCIe only.** BW=50 GB/s. Grace C2C (~900 GB/s) makes every row hidden trivially;
  this phase is the harder PCIe case the project targets.

## 7. Bottom line

On real weights, MoE decode routing is skewed enough (top-25% experts = 48-60% of
traffic) and temporally local enough (k=4 cycle touches only ~15-29% of the pool at
low batch) that a **DRAM-offloaded expert draft is PCIe-viable**: with a 12.5-25%
verify-warmed cache, the per-cycle streamed bytes fall **below** the 50 GB/s
hideable budget at the draft's native low batch, holding GPU memory to the bf16
verify shard + ~2-5 GB. Streaming is fully hidden where it matters; beyond that the
skip-cold valve degrades acceptance gracefully without ever stalling or losing
correctness. The earlier "fetch can't be hidden on PCIe" wall holds only for
on-demand full-set swaps -- not for this reuse-amortized, cache-backed partial stream.
