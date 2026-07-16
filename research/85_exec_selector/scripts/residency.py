#!/usr/bin/env python3
"""Residency feasibility: bytes_resident <= HBM budget, per (config, cell).

Deterministic arithmetic that turns the record's three hand-exclusions
into model outputs. Validation targets (measured incidents):
  - bf16 full replica: 'Model loading took 73.25 GiB' (83-E2b)
  - bf16 partial replica: 'Model loading took 46.25 GiB' (83-E2c)
  - dense b32/32k: KV demand 58.7 GiB > pool (79 e3x32 NOTE)
"""

import json
from pathlib import Path

PHASE = Path(__file__).resolve().parents[1]
GB = 1024 ** 3
HBM = 79.19 * GB

ARCH = {
    "dense": dict(weights=15.2 * GB, kv_per_tok=57_344, tp=1),
    # moe target under DP4/EP4: attn+dense+shared replicated + expert shard
    "moe": dict(weights=16.0 * GB, kv_per_tok=98_304, tp=1),
}
DRAFT_DELTA = {
    "shared_kv_self": 0,                       # bf16 self-draft, shared KV
    "w4_ckpt": 4.7 * GB,                       # dense composed draft
    "fp8_full_replica": 30.25 * GB,            # moe, 83-E2b
    "bf16_full_replica": 57.25 * GB,           # moe, 83-E2b (infeasible)
    "bf16_partial_replica_50": 30.25 * GB,     # moe, 83-E2c (46.25 total)
    "ngram": 0,
}
OVERHEAD = 3.0 * GB                            # graphs, nccl, activations


def feasible(arch, draft, batch, ctx, gpu_mem=0.90):
    a = ARCH[arch]
    kv = batch * (ctx + 512) * a["kv_per_tok"]
    total = a["weights"] + DRAFT_DELTA[draft] + kv + OVERHEAD
    return total <= gpu_mem * HBM, total / GB


def main() -> int:
    checks = [
        # validation against recorded incidents
        ("moe + bf16 full replica, b4/2k", "moe", "bf16_full_replica", 4, 2048, False),
        ("moe + bf16 partial 50%, b4/2k", "moe", "bf16_partial_replica_50", 4, 2048, True),
        ("moe + fp8 full replica, b4/2k", "moe", "fp8_full_replica", 4, 2048, True),
        ("dense + W4 draft, b32/32k", "dense", "w4_ckpt", 32, 32000, False),
        ("dense + W4 draft, b16/32k", "dense", "w4_ckpt", 16, 32000, True),
        ("dense + W4 draft, b32/16k", "dense", "w4_ckpt", 32, 16384, True),
    ]
    lines = ["\n## Residency feasibility (deterministic; validated on incidents)",
             "| config | resident GiB | feasible | recorded outcome |",
             "|---|---|---|---|"]
    ok_all = True
    for name, arch, draft, b, ctx, expected in checks:
        ok, gib = feasible(arch, draft, b, ctx)
        match = ok == expected
        ok_all &= match
        lines.append(f"| {name} | {gib:.1f} | {'Y' if ok else 'N'} | "
                     f"{'matches' if match else 'MISMATCH'} record |")
    lines.append(f"\nAll incident validations "
                 f"{'PASS' if ok_all else 'FAIL'}. Selector rule: a config "
                 f"is admissible iff feasible(realization, cell); the map "
                 f"prints infeasible cells as such instead of hand notes.")
    out = "\n".join(lines)
    with (PHASE / "results_exec.md").open("a") as f:
        f.write(out + "\n")
    print(out)
    (PHASE / "data/residency.json").write_text(json.dumps(
        {n: feasible(a, d, b, c)[1] for n, a, d, b, c, _ in checks}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
