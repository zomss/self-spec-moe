# Phase 51 results — h107↔h106 fabric microbench

**Setup.** h107 + h106, each 8x H100-80GB NVSwitch + 8x ConnectX-7 400G rails
(RoCE v2, GID index 3), GPUDirect RDMA (`nvidia_peermem`+`gdrdrv`). NCCL 2.28.9
via torch 2.11 (`.venv`), `NCCL_IB_HCA=^mlx5_8`, bootstrap over frontend
`ens14np0`. NCCL log confirms all 8 rails per node, cross-node channels
`via NET/IB/GDRDMA` (no sockets, no fallback). Logs in `logs/`.

## 1. Raw RDMA (perftest, host memory, rail mlx5_0)

- `ib_write_bw -a`: peaks ~386 Gb/s @16 KB (single QP; large-size dips are
  CPU-freq/NUMA artifacts of host-memory perftest, not the link).
- `ib_write_lat -a`: **2.4-3.6 us** across 2 B-16 KB. Excellent fabric.

## 2. NCCL per-collective latency, 2 nodes x 8 GPUs (world=16)

Back-to-back same-stream ops, CUDA-event timed, max across ranks
(`scripts/bench_nccl_a2a.py`).

| op | bytes/rank | us/op |
|---|---:|---:|
| all_to_all_single | 16 KB | 113.1 |
| all_to_all_single | 64 KB | 68.7 |
| all_to_all_single | 256 KB | 51.3 |
| all_to_all_single | 1 MB | 68.2 |
| all_to_all_single | 4 MB | 133.7 |
| all_to_all_single | 16 MB | 471.5 |
| all_to_all_single | 64 MB | 1136.3 |
| all_reduce | 8 B | 38.5 |
| all_reduce | 4 KB | 38.6 |
| all_reduce | 1 MB | 65.3 |

Floor ~51 us (256 KB); the 16 KB point (113 us) is *above* the 64-256 KB ones
(NCCL proto/channel scheduling at tiny payloads), so decode-sized collectives
sit at **~70-115 us**. Small all_reduce (the DP sync barrier, cf. Phase 45
residual) is ~38 us/op cross-node.

## 3. Gate reading (Phase 19 / Phase 47)

- **Phase 19 go-gate (>=200 us/coll at decode sizes): NOT met** on this fabric
  at Phase-42's 16-64 KB/rank decode payloads (~70-115 us). It IS met at
  >=16 MB payloads, i.e. prefill/very-large-batch only.
- **Phase 47 mapping:** at ~100 us/coll the measured-shielded win curve gives
  **~1.37x (b32) / ~1.19x (b64)**; at 250 us, 1.61/1.39. Implied f for
  Qwen3-30B EP16 (96 colls/step x 70-115 us ~= 7-11 ms A2A vs ~15 ms compute
  floor) is **f ~= 0.3-0.4** -> cost-model speedup ~1.2-1.3x at K=2,
  accept~2.9. So this 2-node fabric lands in the **modest-win band
  (~1.2-1.4x)**, not the f=0.6-0.8 / 1.6-2.1x projection band — the "fabric
  too good" risk called out before measuring is real: 8x400G rail-optimized
  RoCE with GDRDMA is near the best case for the *baseline*.
- Levers that raise f on real hardware (unchanged from Phase 47 discussion):
  deeper/larger MoE (more collectives per step — DeepSeek-scale has ~28 ms
  A2A/step in public DeepEP numbers), wider EP, lower per-rank compute. The
  honest headline will be a win *curve* vs model scale/fabric, with this
  fabric as the strong-baseline point.

## Caveats

- `all_to_all_single` is a dense uniform exchange; vLLM EP dispatch/combine is
  all-to-all-v with routing imbalance (worse) but DeepEP low-latency kernels
  target lower per-coll latency (better). Treat 70-115 us as the NCCL
  operating point, to be re-measured in-engine.
- perftest is host-memory (no CUDA build); GPUDirect was validated via NCCL.

## Next artifact (Phase 52)

First real 2-node vLLM run: Qwen3-30B, DP/EP=16, Phase-47 spec stack (FP8
full-replica comm-free draft, K=2, PIECEWISE), measure real f, real exposed
per-collective A2A in-engine, and end-to-end spec vs no-spec tokens/s.
Prediction to test: **~1.2-1.4x at b32-64**. Then a DeepSeek-scale
shared-expert model to move up the f curve.
