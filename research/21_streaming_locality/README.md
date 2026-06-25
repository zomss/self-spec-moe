# Phase 21: Union-over-k Reuse and Cache-Hit Curves for PCIe-Streamed Experts

Source: Phase 20 (injection emulation of inter-node A2A), Phase 10 (temporal
locality cache; capture reused here), Phase 18 (quantized-expert draft beta).

## Objective

The PCIe-viable expert-offload draft keeps GPU memory at ~the bf16 verify shard and
streams quantized experts from DRAM during the draft, hidden behind compute. From
the design analysis, overlap only hides the load up to the budget `k * T_step * BW`;
the binding question is **how few bytes must actually be streamed per draft cycle**
once temporal reuse and a resident cache remove repeats. Two numbers decide it:

```
1. union-over-k reuse: distinct experts/layer in a k-token (x B-seq) draft cycle.
2. cache-hit curve:     fraction of routing served by a size-C resident cache.
```

Then: `streamed_bytes(C,k,B) = union(k,B) * (1 - hit(C)) * L * expert_bytes`, and the
draft is PCIe-hidden iff `streamed_bytes <= k * T_step * BW`.

## Method

`routing_locality.py` captures per-layer per-position true top-k experts from a
**real-weight** decode trace (mandatory: dummy weights route uniformly), saves the
raw routing to `data/<tag>_routing.npz`, then computes union-over-k, the four
cache-hit curves (verify-warmed LRU / per-request static / global static / random),
routing skew, and the PCIe-budget synthesis. Analysis is pure-CPU and re-runs from
the saved trace with `--reuse`.

```bash
V=/data/smcho/self-spec-moe/.venv/bin/python
CUDA_VISIBLE_DEVICES=0 $V routing_locality.py --model Qwen/Qwen3-30B-A3B \
  --tag qwen3_30b --local-files-only --prompts-per-bucket 4 --gen-tokens 160
CUDA_VISIBLE_DEVICES=1 $V routing_locality.py --model openai/gpt-oss-20b \
  --tag gptoss_20b --local-files-only --cache-list 0,4,6,8,12,16,24
# re-analyze (no GPU), e.g. sweep the PCIe params:
$V routing_locality.py --tag qwen3_30b --reuse --pcie-gbps 50 --tstep-ms 9
```

## Decision criterion

PCIe-viable if, at the draft's native low batch, a *modest* cache (<= 25% of experts)
brings `fit = streamed/hideable <= 1` for useful `k` -- i.e. the stream is fully
hidden behind draft compute on a ~50 GB/s link, while extra HBM stays well below the
full-replicate cost.

## Result (see `results_streaming_locality.md`)

**PCIe-viable at low batch.** Real routing is skewed (Qwen3 top-25% experts = 60% of
traffic) and temporally local (B=1, k=4 cycle touches only ~20/128 experts/layer).
With a 12.5-25% verify-warmed cache the per-cycle streamed bytes fall **below** the
50 GB/s hideable budget at low batch (Qwen3 C=16, k=4: fit 0.52; C=32: 0.27),
holding GPU memory to the bf16 shard + ~2-5 GB (vs +14.5 GB to replicate). Mid/high
batch needs a larger cache or the skip-cold valve; either way no stall, no
correctness loss. Confirmed on both Qwen3-30B-A3B (E=128) and GPT-OSS-20B (E=32),
two architectures and quant regimes.

## Next artifact

The PCIe budget assumes a draft `T_step`; the real lockstep draft/verify scheduler
(or a 2-node IB run, Phase 19) would measure the actual draft step and DBO residual
to pin `f`. This phase supplies the streamed-bytes and cache-size operating point
that scheduler should target.
