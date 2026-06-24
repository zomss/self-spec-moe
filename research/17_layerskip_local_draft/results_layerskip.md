# Results: Layer-Skip + Local-Routing Draft -- Acceptance vs Cost

Date: 2026-06-24

Tests whether making the draft cheaper via **layer-skip** (skip a band of middle
decoder layers, identity passthrough) on top of **local routing** (M=E/2 experts)
buys throughput by enabling longer drafts. Qwen3-30B-A3B, one-step acceptance
proxy (16 prompts). Cost from Phase 16 pinned numbers
(`T_draft/T_full = ((L-S)/L) * ((1-phi_moe)+r_moe*phi_moe)`, phi_moe=0.7).
Throughput gain at verify creep ~1.15x.

## Pure layer-skip (full routing, M=128)

| skip layers | skip frac | sampled acc | T_draft/T_full | gain k=2 | k=4 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.00 | 1.000 | 1.000 | 0.95 | 0.97 |
| 6 | 0.125 | 0.729 | 0.875 | 0.78 | 0.63 |
| 12 | 0.25 | 0.556 | 0.750 | 0.70 | 0.51 |
| 16 | 0.33 | 0.421 | 0.667 | 0.64 | 0.45 |
| 24 | 0.50 | 0.093 | 0.500 | 0.51 | 0.35 |

## Local routing (M=E/2) + layer-skip

| skip layers | skip frac | sampled acc | T_draft/T_full | gain k=2 | k=4 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.00 | **0.831** | 0.685 | **1.00** | 0.92 |
| 6 | 0.125 | 0.682 | 0.599 | 0.92 | 0.76 |
| 12 | 0.25 | 0.634 | 0.514 | 0.94 | 0.77 |
| 16 | 0.33 | 0.395 | 0.457 | 0.75 | 0.55 |
| 24 | 0.50 | 0.095 | 0.343 | 0.60 | 0.44 |

## Finding: layer-skip is a bad trade

**The best throughput across every configuration is `~1.00` at `skip=0` (pure local
routing). Every amount of layer-skip makes it worse.** The reason is a slope
mismatch:

```text
Cost drops ~linearly with skipped layers ((L-S)/L).
Acceptance drops SUPER-linearly: each skipped layer removes a real contribution to
the residual stream, which verify must reject. Contiguous middle-band skip is
especially damaging (the hidden state stops evolving for a quarter of the network).
```

Concretely, going local skip 0 -> 12 (1/4 of layers) cuts cost 0.685 -> 0.514
(-0.17) but acceptance 0.831 -> 0.634 (-0.20); the acceptance loss outweighs the
cost saving, so the gain falls (1.00 -> 0.94). Even 1/8 skip already loses. By half
the layers, acceptance collapses to ~0.09 -- the draft is broken.

This is the same lesson the whole project keeps hitting from a new angle: **the
draft is the same model, and every layer matters**, so it cannot be cheaply
approximated without destroying acceptance. Layer-skip saves compute linearly but
costs acceptance super-linearly. The "cheap draft enables long k" hope fails
because a cheap-via-skip draft has too-low acceptance for long drafts to pay off.

## Caveat

This is **naive contiguous middle-band** skip. A distributed skip (every other
layer), or an importance-/learned skip (SWIFT / LayerSkip style), would skip the
genuinely-redundant layers and might preserve acceptance better at a given skip
fraction. But the measured slope is steep (1/8 skip already costs ~0.15
acceptance), so a smarter skip would have to be dramatically better to flip the
trade. One-step proxy; multi-token would be worse.

## Conclusion

For Qwen3-30B, **layer-skip + local routing does not beat pure local routing on
throughput** -- it makes it worse. The training-free "cheap draft" lever (Section 9A
of the final report) fails with naive layer-skip. The remaining throughput levers
are the ones that do not approximate the draft's quality: **quantized experts**
(memory) and a **trained draft head** (cheap and high-acceptance by construction).
