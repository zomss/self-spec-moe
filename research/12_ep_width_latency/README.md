# Phase 12: Exact EP-Width Decode Latency (EP2xDP2 vs EP4xDP1)

Source phase: [`../11_intranode_ep_draft`](../11_intranode_ep_draft)

## Objective

Replace the Phase 11 analytical timing model with a **real** vLLM measurement:
how much does the expert all-to-all cost as the EP group grows, on this hardware?
Compare the two layouts on the same 4 GPUs:

- **EP4xDP1**: one engine, `TP=4 + enable_expert_parallel` -> EP=4, all-to-all over 4.
- **EP2xDP2**: two independent engines, each `TP=2 + EP` -> EP=2, all-to-all over 2,
  global batch split in half.

In this vLLM, `ep_size = dp_size * tp_size` (the full world), so a single
EP-enabled engine on 4 GPUs is always EP=4; EP=2 on 4 GPUs requires two
independent EP=2 engines. This maps to the hierarchical draft: draft = small
(intra-node) EP, verify = large EP.

## Setup

| Item | Value |
| --- | --- |
| Model | `Qwen/Qwen3-30B-A3B`, `--load-format dummy` (timing only) |
| GPUs | 2,3,4,5 (single node, NVLink/NVSwitch) |
| Decode shape | input_len=4, output_len=64, ms/token = total/64 |
| Metric | median over 10 iters, 3 warmup |

## Key caveat

This is **single-node NVLink only**. It measures the intra-node EP-width effect.
The inter-node all-to-all -- the cost the hierarchical draft would actually
avoid -- cannot be produced here (forced-slow NCCL crashes; no IB testbed). So
this bounds the *intra-node* component; the inter-node component stays unmeasured.

## Artifacts

| File | Purpose |
| --- | --- |
| `bench_engine_latency.py` | Single-engine decode latency sweep |
| `data/ep4dp1.json`, `data/ep2.json` | Raw ms/token by batch |
| `results_ep_width_latency.md` | Comparison + interpretation |
