#!/usr/bin/env python3
"""Measure the real PCIe link cost on this server (vs NVLink), to calibrate the
comm-bound operating point that the integrated path (NCCL_P2P_DISABLE) can't produce.

The all-to-all moves token activations between GPUs. On a no-NVLink box that goes over
PCIe -- either direct PCIe-P2P (one hop, optimistic) or host-staged GPU->host->GPU
(two hops, pessimistic). We can't force PCIe-P2P here (cudaMemcpyPeer uses NVLink when
present), but we CAN measure: one-way PCIe (D2H/H2D, the direct-P2P proxy), host-staged
round-trip (pessimistic), and NVLink P2P (the fast baseline). Reports bandwidth and
small-message latency per size -> the real PCIe/NVLink gap -> where f actually lands.
"""

from __future__ import annotations

import time

import torch


def timed(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters  # s/call


def main():
    assert torch.cuda.device_count() >= 2, "need >=2 GPUs"
    sizes = [4 << 10, 64 << 10, 1 << 20, 4 << 20, 16 << 20, 64 << 20, 256 << 20]
    print(f"{'bytes':>10} {'D2H GB/s':>9} {'H2D GB/s':>9} {'staged us':>10} "
          f"{'stagedGB/s':>10} {'NVLink GB/s':>11} {'NVL us':>8}")
    rows = []
    for nb in sizes:
        n = nb // 2  # bf16 elements
        g0 = torch.empty(n, dtype=torch.bfloat16, device="cuda:0")
        g0b = torch.empty(n, dtype=torch.bfloat16, device="cuda:0")
        g1 = torch.empty(n, dtype=torch.bfloat16, device="cuda:1")
        h = torch.empty(n, dtype=torch.bfloat16, pin_memory=True)
        it = 100 if nb <= (1 << 20) else 30
        wu = 20

        d2h = timed(lambda: h.copy_(g0, non_blocking=True), it, wu)
        h2d = timed(lambda: g0b.copy_(h, non_blocking=True), it, wu)

        def staged():
            h.copy_(g0, non_blocking=True)
            g0b.copy_(h, non_blocking=True)
        st = timed(staged, it, wu)

        def p2p():
            g1.copy_(g0, non_blocking=True)
        nv = timed(p2p, it, wu)

        row = {
            "bytes": nb,
            "d2h_gbps": round(nb / d2h / 1e9, 1),
            "h2d_gbps": round(nb / h2d / 1e9, 1),
            "staged_us": round(st * 1e6, 1),
            "staged_gbps": round(nb / st / 1e9, 1),
            "nvlink_gbps": round(nb / nv / 1e9, 1),
            "nvlink_us": round(nv * 1e6, 1),
        }
        rows.append(row)
        print(f"{nb:>10} {row['d2h_gbps']:>9} {row['h2d_gbps']:>9} "
              f"{row['staged_us']:>10} {row['staged_gbps']:>10} "
              f"{row['nvlink_gbps']:>11} {row['nvlink_us']:>8}")

    # asymptotic (largest size) ratios
    big = rows[-1]
    print(f"\nAsymptotic BW: NVLink {big['nvlink_gbps']} GB/s, "
          f"PCIe 1-hop(D2H) {big['d2h_gbps']} GB/s, "
          f"PCIe staged {big['staged_gbps']} GB/s")
    print(f"NVLink/PCIe-1hop ratio ~ {big['nvlink_gbps']/max(big['d2h_gbps'],1e-9):.1f}x; "
          f"NVLink/PCIe-staged ~ {big['nvlink_gbps']/max(big['staged_gbps'],1e-9):.1f}x")
    import json
    from pathlib import Path
    Path("data").mkdir(exist_ok=True)
    Path("data/pcie_link.json").write_text(json.dumps({"rows": rows}, indent=2))
    print("[saved] data/pcie_link.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
