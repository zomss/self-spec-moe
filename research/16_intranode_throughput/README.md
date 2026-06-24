# Phase 16: Intra-Node Throughput via Reduced Expert Activation

Source: [`../13_affinity_placement_gated_draft`](../13_affinity_placement_gated_draft),
[`../15_replication_phi_envelope`](../15_replication_phi_envelope)

## Objective

The latency story is intra-node-dead (Phase 12). Test a different axis: at
memory-bound (moderate-high batch) decode, a local-only draft activates fewer
unique experts -> reads fewer expert weights -> cheaper draft step, with **no
communication involved**. This is a *throughput* mechanism, fully intra-node.

## Method

Drive vLLM's real `fused_experts` Triton kernel with routing restricted to `M`
distinct experts; sweep `M` and batch. Per-MoE-layer time. Combine the measured
`M=E/2` vs `M=E` ratio with acceptance (Phase 13/15) into a throughput envelope.

```bash
.venv/bin/python research/16_intranode_throughput/bench_moe_experts.py \
    --num-experts 128 --top-k 8 --hidden 2048 --intermediate 768 \
    --batches 16,64,256,1024,4096 --active-experts 8,16,32,64,96,128 \
    --output-json research/16_intranode_throughput/data/qwen3_moe.json
```

## Result

See `results_throughput.md`. Mechanism confirmed: `phi=0.5` draft costs ~0.54-0.71x
the MoE FFN at serving batch (B=64-1024). But end-to-end gain is modest (~1.05-1.14x)
and conditional on MoE-dominated decode + short draft + high acceptance. Real,
communication-free, intra-node -- but small, and overlaps SS-MoE's mechanism.
