# Phase 11: Hierarchical-EP Draft Feasibility

Source phase: [`../10_temporal_locality_cache`](../10_temporal_locality_cache)

## Objective

Before committing to the intra-node-EP draft design, verify the two quantities it
needs: (1) is the draft step short enough for speculation, and (2) is acceptance
high enough? Idea: draft with **intra-node EP** (NVLink all-to-all only), verify
with **full EP** (inter-node IB all-to-all), so the draft avoids the expensive
inter-node hop while keeping a node-sized local expert pool.

## What is measurable here vs modeled

- **Acceptance:** measurable now. Intra-node draft acceptance = local routing with
  `local = E/nodes` (a node's expert shard). Drawn from Phase 07 (realistic
  contiguous placement) and Phase 09/10 (with a per-node high-coverage cache).
- **Intra-node all-to-all latency:** measured (`bench_alltoall.py`, NVLink).
- **Per-token compute floor:** measured (`bench_decode_compute.py`, single GPU,
  HF eager -- an overestimate vs optimized serving; flagged).
- **Inter-node all-to-all latency:** modeled. Forcing a slow NCCL transport
  (`NCCL_P2P_DISABLE` / `SHM_DISABLE`) crashes here (`ncclRemoteError` / SIGABRT),
  the same instability Phase 02 found, so inter-node uses an analytical IB model.

## Method note

All-to-all is paid **per MoE layer x 2** (dispatch + combine), so a decode step
has `2 * num_layers` collectives. That is what turns a ~30 us per-collective
latency into milliseconds per step and makes the inter-node fraction matter.

## Artifacts

| File | Purpose |
| --- | --- |
| `bench_alltoall.py` | NCCL all-to-all latency sweep (run per transport) |
| `bench_decode_compute.py` | Per-token decode compute floor (single GPU) |
| `feasibility_envelope.py` | Combine measured + modeled into speedup envelope |
| `data/`, `logs/` | Raw measurements |
| `results_feasibility.md` | Verdict on both questions |

## Commands

```bash
.venv/bin/python research/11_intranode_ep_draft/bench_alltoall.py \
    --label nvlink --out research/11_intranode_ep_draft/data/a2a_nvlink.json
.venv/bin/python research/11_intranode_ep_draft/bench_decode_compute.py \
    --model Qwen/Qwen3-30B-A3B --out research/11_intranode_ep_draft/data/compute_qwen3.json
.venv/bin/python research/11_intranode_ep_draft/feasibility_envelope.py
```
