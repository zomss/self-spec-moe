#!/usr/bin/env python3
"""Measure fused-MoE expert-FFN time vs (#active experts M, batch B).

Drives vLLM's real fused_experts Triton kernel. Restricting routing to M distinct
experts loads only those M expert weights -> at the memory-bound decode regime the
step gets cheaper. This is the throughput mechanism of a local-only draft (the
draft activates fewer experts than full routing). Per-layer time; a step is
num_layers of these.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def parse_int_list(raw: str) -> list[int]:
    return [int(v) for v in raw.split(",") if v.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=2048)        # Qwen3-30B-A3B
    ap.add_argument("--intermediate", type=int, default=768)   # moe_intermediate_size
    ap.add_argument("--num-experts", type=int, default=128)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--batches", type=parse_int_list, default=[1, 4, 16, 64, 256, 1024])
    ap.add_argument("--active-experts", type=parse_int_list, default=None)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--label", default="qwen3")
    ap.add_argument("--output-json", type=Path, required=True)
    a = ap.parse_args()

    from vllm.model_executor.layers.fused_moe import fused_experts

    dev = torch.device(a.device)
    E, H, I, K = a.num_experts, a.hidden, a.intermediate, a.top_k
    # vLLM convention: w1 = [E, 2*I, H] (gate+up), w2 = [E, H, I] (down)
    w1 = torch.randn(E, 2 * I, H, dtype=torch.bfloat16, device=dev) * 0.02
    w2 = torch.randn(E, H, I, dtype=torch.bfloat16, device=dev) * 0.02
    Ms = a.active_experts or sorted(set([K, E // 8, E // 4, E // 2, 3 * E // 4, E]))
    Ms = [m for m in Ms if K <= m <= E]
    gen = torch.Generator(device="cpu").manual_seed(0)
    print(f"[{a.label}] H={H} I={I} E={E} K={K}  M={Ms}")

    def time_call(B, M):
        hs = torch.randn(B, H, dtype=torch.bfloat16, device=dev)
        # route each token to K distinct experts drawn from the first M experts
        ids = torch.empty(B, K, dtype=torch.int32, device=dev)
        for t in range(B):
            perm = torch.randperm(M, generator=gen)[:K]
            ids[t] = perm.to(torch.int32)
        tw = torch.softmax(torch.randn(B, K, generator=gen).to(dev).float(), dim=-1)
        for _ in range(a.warmup):
            fused_experts(hs, w1, w2, tw, ids, global_num_experts=E)
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(a.iters):
            fused_experts(hs, w1, w2, tw, ids, global_num_experts=E)
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / a.iters  # ms per layer call

    rows = []
    for B in a.batches:
        rec = {"batch": B}
        for M in Ms:
            ms = time_call(B, M)
            rec[f"M{M}"] = round(ms, 4)
        # ratio of M=E/2 vs M=E (the phi=0.5 draft vs full)
        if f"M{E//2}" in rec and f"M{E}" in rec:
            rec["ratio_half_vs_full"] = round(rec[f"M{E//2}"] / rec[f"M{E}"], 3)
        rows.append(rec)
        cols = "  ".join(f"M{M}={rec[f'M{M}']:.3f}" for M in Ms)
        print(f"B={B:>5}: {cols}  | half/full={rec.get('ratio_half_vs_full')}")

    a.output_json.parent.mkdir(parents=True, exist_ok=True)
    a.output_json.write_text(
        json.dumps({"label": a.label, "H": H, "I": I, "E": E, "K": K,
                    "active_experts": Ms, "rows": rows}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
